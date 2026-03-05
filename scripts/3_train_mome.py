"""
[Step 3] MoME v2.8.3 训练脚本 (全面优化版)
核心改进：
1. 动态模态屏蔽 (Modal Dropout)：增强单模态补位能力，防止专家霸权。
2. 真实质量注入：从数据包中动态加载 quality_2d，实现环境感知路由。
3. 余弦退火调度：优化学习率衰减轨迹，提升收敛精度。
4. TensorBoard 深度监控：全维度追踪 Loss 分解与专家话语权变动。
5. 防作弊增强：维持物理维度屏蔽，并加入 Precision 监控。
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


class RoadDatasetV28(Dataset):
    def __init__(self, config):
        self.files = sorted(list(Path(config["paths"]["output_dir"]).glob("*.npz")))
        self.max_patches = config["geometry"].get("max_patches", 81)
        self.q2_default = config["inference"].get("q2_default", 0.8)
        self.manual_labels = {}
        m_path = Path(config["paths"]["manual_label_path"])
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                self.manual_labels = json.load(f)
            print(f"✅ 成功加载 {len(self.manual_labels)} 帧人工真值。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = np.load(path, allow_pickle=True)
            img_key = path.stem.split("_")[-1] + ".jpg"

            phys_raw = data["phys_8d"]
            geom_raw = data["deep_512d"]
            tex_raw = data["deep_2d_768d"]
            meta_raw = data["meta"]  # [label, q_geo]
            # 动态读取图像质量分
            q2_val = data.get("quality_2d", np.array([self.q2_default]))[0]
            n_real = phys_raw.shape[0]

            # 容器初始化
            phys = np.zeros((self.max_patches, 8), dtype=np.float32)
            geom = np.zeros((self.max_patches, 384), dtype=np.float32)
            meta = np.zeros((self.max_patches, 2), dtype=np.float32)
            v_gt = np.zeros(self.max_patches, dtype=np.float32)
            v_mask = np.zeros(self.max_patches, dtype=np.float32)

            # 数据填充
            limit = min(n_real, self.max_patches)
            phys[:limit] = phys_raw[:limit]
            geom[:limit] = geom_raw[:limit]
            meta[:limit] = meta_raw[:limit]

            # 匹配标注
            if img_key in self.manual_labels:
                labels = np.array(self.manual_labels[img_key])
                l_limit = min(len(labels), self.max_patches)
                v_gt[:l_limit] = labels[:l_limit]
                v_mask[:l_limit] = 1.0

            return (
                torch.from_numpy(phys),
                torch.from_numpy(geom),
                torch.from_numpy(tex_raw).float(),
                torch.from_numpy(meta),
                torch.from_numpy(v_gt),
                torch.from_numpy(v_mask),
                torch.tensor([q2_val], dtype=torch.float32),
            )
        except Exception:
            return self.__getitem__((idx + 1) % len(self.files))


def train():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mome_model(cfg).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-2
    )
    # 引入学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    writer = SummaryWriter(log_dir=cfg["paths"]["log_dir"])

    full_ds = RoadDatasetV28(cfg)
    train_size = int(0.85 * len(full_ds))
    val_ds, train_ds = random_split(full_ds, [len(full_ds) - train_size, train_size])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    print(f"🚀 v2.8.3 深度协同训练启动 | 样本数: {len(full_ds)}")
    best_f1 = 0
    mask_p = cfg["train"].get("modal_mask_prob", 0.3)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        m = {
            "total": 0,
            "main": 0,
            "vis": 0,
            "ale": 0,
            "weights": torch.zeros(4).to(device),
            "steps": 0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for phys, geom, tex, meta, v_gt, v_mask, q2_val in pbar:
            phys, geom, tex, meta, v_gt, v_mask, q2_val = [
                x.to(device) for x in [phys, geom, tex, meta, v_gt, v_mask, q2_val]
            ]

            # --- [增强：防作弊与模态屏蔽] ---
            phys[:, :, 0] = 0.0  # 屏蔽 max_dz
            if torch.rand(1) < mask_p:  # 随机屏蔽 2D 输入，强制 3D 独立学习
                tex = torch.zeros_like(tex)
                q2_val = torch.zeros_like(q2_val)

            # 构造质量向量 [B, N, 2]
            q_geo = meta[:, :, 1:2]
            q_img = q2_val.unsqueeze(1).expand(-1, q_geo.size(1), -1)
            q_vec = torch.cat([q_geo, q_img], dim=-1)

            # 前向传播
            logits, weights, exp_logits, uncerts = model(phys, geom, tex, q_vec)
            targets = meta[:, :, 0]
            conf = meta[:, :, 1]

            # 1. 损失函数拆解
            main_loss = (criterion(logits, targets) * conf).mean()

            ale_loss = 0
            for i, idx in enumerate([1, 2, 3]):  # Geom, Tex, Synergy
                s = uncerts[:, :, i]
                bce = criterion(exp_logits[:, :, idx], targets)
                ale_loss += (torch.exp(-s) * bce + 0.5 * s).mean()

            # 2. 视觉真值引导 (梯度唤醒)
            vis_loss = (criterion(exp_logits[:, :, 2], v_gt) * v_mask).sum() / (
                v_mask.sum() + 1e-6
            )

            # 3. 动态聚合
            total_loss = (
                main_loss + 0.15 * ale_loss + cfg["train"]["lambda_vis"] * vis_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # 梯度裁剪增加稳定性
            optimizer.step()

            # 统计
            m["total"] += total_loss.item()
            m["vis"] += vis_loss.item()
            m["ale"] += ale_loss.item()
            m["weights"] += weights.mean(dim=[0, 1]).detach()
            m["steps"] += 1
            pbar.set_postfix(
                {"L": f"{total_loss.item():.3f}", "V": f"{vis_loss.item():.3f}"}
            )

        scheduler.step()

        # --- 验证与日志 ---
        avg_w = (m["weights"] / m["steps"]).cpu().numpy()
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for phys, geom, tex, meta, _, _, q2 in val_loader:
                phys, geom, tex, meta, q2 = [
                    x.to(device) for x in [phys, geom, tex, meta, q2]
                ]
                phys[:, :, 0] = 0.0
                q_vec = torch.cat(
                    [meta[:, :, 1:2], q2.unsqueeze(1).expand(-1, 81, -1)], dim=-1
                )
                logits, _, _, _ = model(phys, geom, tex, q_vec)
                mask = meta[:, :, 1] > 0
                all_preds.extend(
                    (torch.sigmoid(logits)[mask] > 0.5).int().cpu().numpy()
                )
                all_gts.extend(meta[:, :, 0][mask].int().cpu().numpy())

        f1 = f1_score(all_gts, all_preds, zero_division=0)
        rec = recall_score(all_gts, all_preds, zero_division=0)
        prec = precision_score(all_gts, all_preds, zero_division=0)

        # TensorBoard 记录
        writer.add_scalar("Loss/Total", m["total"] / m["steps"], epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Weights/Synergy", avg_w[3], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"\n📊 E{epoch+1} 审计报告: F1={f1:.3f} | Rec={rec:.3f} | Prec={prec:.3f}"
        )
        print(
            f"   权重分布: Phys:{avg_w[0]:.2f} | 3D:{avg_w[1]:.2f} | 2D:{avg_w[2]:.2f} | Syn:{avg_w[3]:.2f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), cfg["paths"]["weights"]["mome_model"])
            print(f"⭐ 发现更优 F1，权重已固化。")

    writer.close()


if __name__ == "__main__":
    train()
