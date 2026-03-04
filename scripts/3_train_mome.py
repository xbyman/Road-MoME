"""
[Step 3] MoME v2.3 训练脚本
核心更新：
1. Quality Jittering: 训练时随机将 30% 样本的图像/几何质量设为 0.05，逼迫门控在“极端逆境”下学习切换。
2. Auxiliary Expert Loss: 引入辅助损失函数。即使某个专家当前权重较低，也强制其学习伪标签，确保专家具备“随时待命”的判别力。
3. Robust Training: 适配 v2.3 模型的 log 空间门控干预。
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
from sklearn.metrics import f1_score

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
            phys = torch.from_numpy(data["phys_8d"]).float()
            geom = torch.from_numpy(data[FEAT_CFG["3d"]["key_name"]]).float()
            tex = torch.from_numpy(data[FEAT_CFG["2d"]["key_name"]]).float()

            q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()
            q_img_val = data.get("quality_2d", np.array([0.8]))[0]
            q_img = torch.full_like(q_geo, q_img_val)
            q_vec = torch.cat([q_geo, q_img], dim=-1)

            meta = torch.from_numpy(data["meta"]).float()
            return phys, geom, tex, q_vec, meta
        except:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_n = max([b[0].shape[0] for b in batch])
    p_phys, p_geom, p_tex, p_qvec, p_meta = [], [], [], [], []
    for phys, geom, tex, qvec, meta in batch:
        n = phys.shape[0]
        pad = max_n - n
        p_phys.append(torch.cat([phys, torch.zeros(pad, phys.shape[1])]))
        p_geom.append(torch.cat([geom, torch.zeros(pad, geom.shape[1])]))
        p_tex.append(tex)
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
    exp_name = TRAIN_CFG.get("experiment_name", "MoME_v2.3_Jittered")
    exp_mgr = ExperimentManager(project_root, experiment_name=exp_name)
    exp_mgr.log_config(config_path)

    model = build_mome_model(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-2)
    pos_w = torch.tensor([TRAIN_CFG["pos_weight"]]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")

    full_ds = RoadDataset(PATH_CFG["output_dir"])
    train_size = int(len(full_ds) * 0.8)
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])

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

            # [v2.3 核心：质量扰动增强]
            # 随机选取 30% 的 Batch 模拟图像全黑/损坏，强迫门控切换至几何专家
            if random.random() < 0.3:
                q_vec[:, :, 1] = 0.05
                tex = tex * 0.0

            # 随机选取 15% 的样本模拟点云稀疏/失效
            if random.random() < 0.15:
                q_vec[:, :, 0] = 0.05
                geom = geom * 0.0

            logits, weights, exp_logits = model(phys, geom, tex, q_vec)

            # 1. 计算主任务损失 (Weighted BCE)
            targets = meta[:, :, 0]
            confidences = meta[:, :, 1]
            raw_loss = criterion(logits, targets)
            main_loss = (raw_loss * confidences).sum() / (confidences.sum() + 1e-6)

            # 2. 计算辅助专家损失 (Auxiliary Loss)
            # 强制 4 个专家即便权重低也要独立学习，权重系数设为 0.3
            aux_loss = 0
            for i in range(4):
                e_loss = criterion(exp_logits[:, :, i], targets)
                aux_loss += (e_loss * confidences).sum() / (confidences.sum() + 1e-6)

            total_loss = main_loss + 0.3 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # 验证与记录
        model.eval()
        val_weights = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                phys, geom, tex, q_vec, meta = [b.to(DEVICE) for b in batch]
                logits, weights, _ = model(phys, geom, tex, q_vec)

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
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        writer.add_scalar("Expert/Synergy_Weight", avg_w[3], epoch)

        if f1 > best_f1:
            best_f1 = f1
            save_path = Path(PATH_CFG["checkpoint_dir"]) / "mome_model_best.pth"
            torch.save(model.state_dict(), save_path)

    writer.close()
    print(f"✨ v2.3 增强版训练完成！最佳 F1: {best_f1:.3f}")


if __name__ == "__main__":
    train()
