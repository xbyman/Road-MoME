"""
[Step 3] MoME 道路异常检测训练脚本 (v3.4 协同适配版)
核心更新：
1. 专家审计：适配 4 专家架构 (Phys, Geom, Tex, Synergy)。
2. 指标监控：在 TensorBoard 中新增 Synergy_Weight 曲线。
3. 协同验证：统计协同专家在异常样本上的话语权提升。
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

# 导入自定义模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model
from scripts.exp_manager import ExperimentManager


# ==================== 环境配置 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), cfg_path


config_dict, config_path = load_config()
TRAIN_CFG = config_dict["train"]
PATH_CFG = config_dict["paths"]
FEAT_CFG = config_dict["features"]
GEO_CFG = config_dict["geometry"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定种子
seed = TRAIN_CFG.get("seed", 42)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 初始化实验管理
exp_mgr = ExperimentManager(
    project_root, experiment_name=TRAIN_CFG.get("experiment_name", "MoME_Synergy_v2")
)
exp_mgr.log_config(config_path)
EXP_DIR = Path(exp_mgr.get_exp_dir())


def get_expected_patch_count(geo):
    step = geo["patch_size"] * (1 - geo["overlap"])
    nx = len(
        np.arange(geo["roi_x"][0], geo["roi_x"][1] - geo["patch_size"] + 0.1, step)
    )
    ny = len(
        np.arange(geo["roi_y"][0], geo["roi_y"][1] - geo["patch_size"] + 0.1, step)
    )
    return nx * ny


MAX_PATCHES = get_expected_patch_count(GEO_CFG)
# =================================================


class RoadDataset(Dataset):
    def __init__(self, npz_dir):
        self.files = sorted(list(Path(npz_dir).glob("*.npz")))
        self.valid_files = [f for f in self.files if f.stat().st_size > 0]
        print(f"📊 发现有效特征包: {len(self.valid_files)}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.valid_files[idx], allow_pickle=True)
            return (
                torch.from_numpy(data["phys_8d"]).float(),
                torch.from_numpy(data[FEAT_CFG["3d"]["key_name"]]).float(),
                torch.from_numpy(data[FEAT_CFG["2d"]["key_name"]]).float(),
                torch.from_numpy(data["meta"]).float(),
            )
        except:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_n = MAX_PATCHES
    p_phys, p_geom, p_tex, p_meta = [], [], [], []
    for phys, geom, tex, meta in batch:
        n = phys.shape[0]
        if n > max_n:
            phys, geom, meta, n = phys[:max_n], geom[:max_n], meta[:max_n], max_n
        pad = max_n - n
        p_phys.append(torch.cat([phys, torch.zeros(pad, phys.shape[1])], dim=0))
        p_geom.append(torch.cat([geom, torch.zeros(pad, geom.shape[1])], dim=0))
        p_tex.append(tex)
        p_meta.append(torch.cat([meta, torch.zeros(pad, meta.shape[1])], dim=0))
    return (
        torch.stack(p_phys),
        torch.stack(p_geom),
        torch.stack(p_tex),
        torch.stack(p_meta),
    )


def train():
    # 1. 模型初始化 (此时加载的是 v2.0 特征协同版模型)
    model = build_mome_model(config_dict).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-2)
    pos_w = torch.tensor([TRAIN_CFG["pos_weight"]]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")

    # 2. 数据准备
    full_ds = RoadDataset(PATH_CFG["output_dir"])
    v_size = int(len(full_ds) * TRAIN_CFG["val_split"])
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - v_size, v_size])
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"], collate_fn=collate_fn
    )

    writer = SummaryWriter(log_dir=str(EXP_DIR / "tboard"))
    best_val_loss = float("inf")
    best_metrics = {}

    print(
        f"\n🚀 协同训练启动 | 架构版本: v2.0 (4专家) | 特征维度: 3D({FEAT_CFG['3d']['input_dim']}) 2D({FEAT_CFG['2d']['input_dim']})"
    )

    for epoch in range(TRAIN_CFG["epochs"]):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if batch is None:
                continue
            phys, geom, tex, meta = [b.to(DEVICE) for b in batch]

            # ModalMask 增强：让协同专家在单一模态缺失时学会寻找残差关联
            if random.random() < TRAIN_CFG.get("modal_mask_prob", 0.2):
                phys = phys * 0.0

            logits, _ = model(phys, geom, tex)
            raw_loss = criterion(logits, meta[:, :, 0])
            weighted_loss = (raw_loss * meta[:, :, 1]).sum() / (
                meta[:, :, 1].sum() + 1e-6
            )

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            train_loss += weighted_loss.item()

        # --- 验证与 4 专家贡献度审计 ---
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        all_weights = []  # 记录专家权重 [B, N, 4]

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                phys, geom, tex, meta = [b.to(DEVICE) for b in batch]
                logits, weights = model(phys, geom, tex)  # weights shape: [B, N, 4]

                v_loss = (criterion(logits, meta[:, :, 0]) * meta[:, :, 1]).sum() / (
                    meta[:, :, 1].sum() + 1e-6
                )
                val_loss += v_loss.item()

                valid_mask = meta[:, :, 1].cpu().numpy() > 0
                all_preds.extend(
                    (torch.sigmoid(logits).cpu().numpy()[valid_mask] > 0.5).astype(int)
                )
                all_labels.extend(meta[:, :, 0].cpu().numpy()[valid_mask].astype(int))
                all_weights.append(weights.cpu().numpy()[valid_mask])

        # 计算评估指标
        avg_train, avg_val = train_loss / len(train_loader), val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # 计算平均专家贡献度 (包含第四专家)
        all_weights_cat = np.concatenate(all_weights, axis=0)
        avg_expert_weights = np.mean(
            all_weights_cat, axis=0
        )  # [w_phys, w_geom, w_tex, w_syn]

        print(
            f"📈 E{epoch+1:02d} | Loss: {avg_val:.4f} | F1: {f1:.3f} | Weights: {np.round(avg_expert_weights, 3)}"
        )

        writer.add_scalar("Loss/Train", avg_train, epoch)
        writer.add_scalar("Loss/Validation", avg_val, epoch)
        writer.add_scalar("Metrics/F1-Score", f1, epoch)
        writer.add_scalar("Expert/Phys_Weight", avg_expert_weights[0], epoch)
        writer.add_scalar("Expert/Geom_Weight", avg_expert_weights[1], epoch)
        writer.add_scalar("Expert/Tex_Weight", avg_expert_weights[2], epoch)
        writer.add_scalar(
            "Expert/Synergy_Weight", avg_expert_weights[3], epoch
        )  # 💡 新增

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), EXP_DIR / "mome_model_best.pth")
            torch.save(
                model.state_dict(),
                Path(PATH_CFG["checkpoint_dir"]) / "mome_model_best.pth",
            )
            best_metrics = {
                "best_val_loss": avg_val,
                "f1_score": f1,
                "avg_w_phys": avg_expert_weights[0],
                "avg_w_geom": avg_expert_weights[1],
                "avg_w_tex": avg_expert_weights[2],
                "avg_w_syn": avg_expert_weights[3],
            }

    exp_mgr.save_results(best_metrics, config_dict)
    writer.close()
    print(f"✨ 协同训练完成！最佳 F1: {best_metrics.get('f1_score'):.3f}")


if __name__ == "__main__":
    train()
