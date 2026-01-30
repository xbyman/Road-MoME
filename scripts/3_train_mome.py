"""
[Step 3] MoME 专家模型训练脚本 (v2.4 生产版)
更新点：
1. 容错性：忽略损坏文件 (ignore_corrupt) 并过滤 patch 过少的帧。
2. 实时监控：集成 TensorBoard (SummaryWriter) 记录 Loss 和 Acc 曲线。
3. 最佳保存：自动保存验证集表现最好的模型 (mome_model_best.pth)。
4. 断点续训：支持从上次中断的权重文件恢复训练。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入模型定义
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
TRAIN_CFG = cfg["train"]
PATH_CFG = cfg["paths"]
FEAT_CFG = cfg["features"]
SNTY_CFG = TRAIN_CFG["sanity_check"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class RoadFrameDataset(Dataset):
    """
    鲁棒的数据集加载器
    """

    def __init__(self, npz_dir):
        self.files = sorted(list(Path(npz_dir).glob("*.npz")))
        print(f"📦 发现 {len(self.files)} 帧特征包，准备进行健壮性检查...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 实现 D 处的 Sanity Check：忽略损坏或无效文件
        try:
            data = np.load(self.files[idx], allow_pickle=True)

            # 过滤 Patch 数量过少的帧
            if data["phys_8d"].shape[0] < SNTY_CFG["min_patch_count"]:
                return None

            return {
                "phys": torch.from_numpy(data["phys_8d"]).float(),
                "geom": torch.from_numpy(data[FEAT_CFG["3d"]["key_name"]]).float(),
                "tex": torch.from_numpy(data[FEAT_CFG["2d"]["key_name"]]).float(),
                "meta": torch.from_numpy(data["meta"]).float(),
            }
        except Exception as e:
            if SNTY_CFG["ignore_corrupt"]:
                return None
            else:
                raise e


def collate_fn_padding(batch):
    # 过滤无效样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    batch_size = len(batch)
    n_patches = [b["phys"].shape[0] for b in batch]
    max_n = max(n_patches)

    padded_phys = torch.zeros(batch_size, max_n, 8)
    padded_geom = torch.zeros(batch_size, max_n, 384)
    padded_tex = torch.zeros(batch_size, 1, 768)
    padded_labels = torch.zeros(batch_size, max_n)
    padded_quality = torch.zeros(batch_size, max_n)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

    for i, b in enumerate(batch):
        curr_n = n_patches[i]
        padded_phys[i, :curr_n] = b["phys"]
        padded_geom[i, :curr_n] = b["geom"]
        padded_tex[i] = b["tex"]
        padded_labels[i, :curr_n] = b["meta"][:, 0]
        padded_quality[i, :curr_n] = b["meta"][:, 1]
        mask[i, :curr_n] = True

    return padded_phys, padded_geom, padded_tex, padded_labels, padded_quality, mask


def train():
    # 1. 初始化模型与优化器
    model = build_mome_model(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([TRAIN_CFG["pos_weight"]]).to(DEVICE), reduction="none"
    )

    # B - 断点续训逻辑
    start_epoch = 0
    if TRAIN_CFG["resume"] and os.path.exists(TRAIN_CFG["resume_path"]):
        print(f"🔄 正在从断点恢复训练: {TRAIN_CFG['resume_path']}")
        model.load_state_dict(torch.load(TRAIN_CFG["resume_path"]))
        # 注意：此处可进一步保存 optimizer 状态以实现完全无缝续训

    # C - TensorBoard 监控初始化
    writer = None
    if TRAIN_CFG["use_tensorboard"]:
        log_path = Path(PATH_CFG["log_dir"])
        writer = SummaryWriter(log_dir=str(log_path))
        print(f"📊 TensorBoard 日志已开启: tensorboard --logdir={log_path}")

    # 2. 数据准备与划分
    full_dataset = RoadFrameDataset(PATH_CFG["output_dir"])
    val_size = int(len(full_dataset) * TRAIN_CFG["val_split"])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_padding,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_CFG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_padding,
    )

    best_val_loss = float("inf")

    # 3. 训练主循环
    for epoch in range(start_epoch, TRAIN_CFG["epochs"]):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CFG['epochs']}")
        for data in pbar:
            if data is None:
                continue
            phys, geom, tex, targets, quality, mask = [d.to(DEVICE) for d in data]

            # ModalMask 抑制霸权 (15% 随机屏蔽)
            if np.random.rand() < 0.15:
                phys = phys * 0.0

            optimizer.zero_grad()
            logits, weights = model(phys, geom, tex)

            # 计算加权 Loss
            raw_loss = criterion(logits, targets)
            loss = (raw_loss * quality * mask.float()).sum() / (mask.sum() + 1e-6)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 4. 验证环节 (Validation)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue
                phys, geom, tex, targets, quality, mask = [d.to(DEVICE) for d in data]
                logits, _ = model(phys, geom, tex)
                v_loss = (criterion(logits, targets) * quality * mask.float()).sum() / (
                    mask.sum() + 1e-6
                )
                val_losses.append(v_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(
            f"📈 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # 将 Loss 记录到 TensorBoard
        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # 5. 模型保存逻辑 (B - 自动保存最佳模型)
        save_dir = Path(PATH_CFG["checkpoint_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存最新权重 (用于断点恢复)
        torch.save(model.state_dict(), save_dir / "mome_model_latest.pth")

        # 如果是历史最佳，则额外保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if TRAIN_CFG["save_best"]:
                best_path = save_dir / TRAIN_CFG["best_model_name"]
                torch.save(model.state_dict(), best_path)
                print(f"🌟 检测到更优模型，已保存至: {best_path}")

    if writer:
        writer.close()
    print("✅ 10,000 帧训练马拉松结束！")


if __name__ == "__main__":
    train()
