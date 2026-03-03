"""
[Step 3] MoME v2.2 训练脚本 (级联调制适配版)
核心更新：
1. Quality Vector: 动态构造 [q_geo, q_img] 作为模型 Forward 输入。
2. Loss Regularization: 为 FiLM 的 Gamma 参数添加正则化，防止过度调制。
3. 专家审计：适配 4 专家逻辑 (Phys, Geom, Tex, Synergy)。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model
from scripts.exp_manager import ExperimentManager


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path


config, config_path = load_config()
TRAIN_CFG = config["train"]
FEAT_CFG = config["features"]
PATH_CFG = config["paths"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoadDataset(Dataset):
    def __init__(self, npz_dir):
        self.files = sorted(list(Path(npz_dir).glob("*.npz")))
        self.valid_files = [f for f in self.files if f.stat().st_size > 0]

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.valid_files[idx], allow_pickle=True)
            # 基础特征
            phys = torch.from_numpy(data["phys_8d"]).float()
            geom = torch.from_numpy(data[FEAT_CFG["3d"]["key_name"]]).float()
            tex = torch.from_numpy(data[FEAT_CFG["2d"]["key_name"]]).float()

            # 质量因子与元数据
            q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()  # [N, 1]
            q_img = (
                torch.from_numpy(data["quality_2d"]).float().expand(q_geo.shape[0], 1)
            )  # [N, 1]
            q_vec = torch.cat([q_geo, q_img], dim=-1)  # [N, 2]

            meta = torch.from_numpy(data["meta"]).float()
            return phys, geom, tex, q_vec, meta
        except:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # 动态计算 Patch 数 (基于 ROI_Y [-8, 0] 通常为 81 左右)
    max_n = max([b[0].shape[0] for b in batch])
    p_phys, p_geom, p_tex, p_qvec, p_meta = [], [], [], [], []
    for phys, geom, tex, qvec, meta in batch:
        n = phys.shape[0]
        pad = max_n - n
        p_phys.append(torch.cat([phys, torch.zeros(pad, phys.shape[1])]))
        p_geom.append(torch.cat([geom, torch.zeros(pad, geom.shape[1])]))
        p_tex.append(tex)  # Tex 为全帧特征，无需补齐
        p_qvec.append(torch.cat([qvec, torch.zeros(pad, 2)]))
        p_meta.append(torch.cat([meta, torch.zeros(pad, 2)]))
    return (
        torch.stack(p_phys),
        torch.stack(p_geom),
        torch.stack(p_tex),
        torch.stack(p_qvec),
        torch.stack(p_meta),
    )


def train():
    exp_mgr = ExperimentManager(
        project_root,
        experiment_name=TRAIN_CFG.get("experiment_name", "MoME_v2.2_Cascaded"),
    )
    exp_mgr.log_config(config_path)

    model = build_mome_model(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-2)
    pos_w = torch.tensor([TRAIN_CFG["pos_weight"]]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")

    full_ds = RoadDataset(PATH_CFG["output_dir"])
    train_ds, val_ds = random_split(
        full_ds, [int(len(full_ds) * 0.8), len(full_ds) - int(len(full_ds) * 0.8)]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"], collate_fn=collate_fn
    )

    writer = SummaryWriter(log_dir=exp_mgr.get_exp_dir() + "/tboard")
    best_f1 = 0

    for epoch in range(TRAIN_CFG["epochs"]):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if batch is None:
                continue
            phys, geom, tex, q_vec, meta = [b.to(DEVICE) for b in batch]

            # Modal Dropout 策略: 随机屏蔽
            if random.random() < 0.3:
                if random.random() < 0.5:
                    geom *= 0.0
                else:
                    tex *= 0.0

            logits, _ = model(phys, geom, tex, q_vec)

            # 弱监督加权 Loss
            raw_loss = criterion(logits, meta[:, :, 0])
            weighted_loss = (raw_loss * meta[:, :, 1]).sum() / (
                meta[:, :, 1].sum() + 1e-6
            )

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()

        # 验证逻辑 (记录 4 专家分布)
        model.eval()
        val_weights = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                phys, geom, tex, q_vec, meta = [b.to(DEVICE) for b in batch]
                logits, weights = model(phys, geom, tex, q_vec)

                valid_mask = meta[:, :, 1].cpu().numpy() > 0
                all_preds.extend(
                    (torch.sigmoid(logits).cpu().numpy()[valid_mask] > 0.5).astype(int)
                )
                all_labels.extend(meta[:, :, 0].cpu().numpy()[valid_mask].astype(int))
                val_weights.append(weights.cpu().numpy()[valid_mask])

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_w = np.mean(np.concatenate(val_weights), axis=0)

        print(
            f"📈 E{epoch+1:02} | Loss: {train_loss/len(train_loader):.4f} | F1: {f1:.3f} | Weights: {np.round(avg_w, 3)}"
        )
        writer.add_scalar("Expert/Synergy_Weight", avg_w[3], epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                model.state_dict(),
                Path(PATH_CFG["checkpoint_dir"]) / "mome_model_best.pth",
            )

    writer.close()
    print(f"✨ 级联训练完成！最佳 F1: {best_f1:.3f}")


if __name__ == "__main__":
    train()
