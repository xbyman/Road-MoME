"""
[Step 1] DINOv2 视觉特征提取与注入脚本 (v2.2 质量感知版)
核心更新：
1. 质量评估：计算 Laplacian Variance (图像锐度) 并存入 'quality_2d'。
2. 批量处理：保持高性能 DataLoader 批处理逻辑。
3. 存储：确保特征键名与 config 对齐。
"""

import os
import sys
import torch
import numpy as np
import yaml
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
PRE_CFG = config.get("preprocess", {})
PATH_CFG = config["paths"]
FEAT_CFG = config["features"]["2d"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = PRE_CFG.get("inference_batch_size", 8)
NUM_WORKERS = PRE_CFG.get("num_workers", 4)
NPZ_DIR = Path(PATH_CFG["output_dir"])
IMG_DIR = Path(PATH_CFG["raw_img_dir"])
OUTPUT_KEY = FEAT_CFG["key_name"]


def calculate_img_quality(img_np):
    """计算拉普拉斯方差作为图像质量分"""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 归一化映射：通常 300 以上为清晰，映射到 [0.1, 1.0]
    q2 = np.clip(score / 300.0, 0.1, 1.0)
    return float(q2)


class ImageBatchDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, npz_path = self.pairs[idx]
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)
            q2 = calculate_img_quality(img_np)
            img_t = self.transform(img_pil)
            return img_t, str(npz_path), torch.tensor([q2], dtype=torch.float32)
        except:
            return torch.zeros(3, 518, 518), "ERROR", torch.tensor([0.1])


def main():
    print("🚀 启动 2D 特征与质量评估流水线 (v2.2)")

    # 1. 建立索引
    img_index = {
        p.stem: p
        for p in IMG_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    }
    npz_files = sorted(list(NPZ_DIR.glob("*.npz")))

    matched_pairs = []
    skipped = 0
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as data:
            if OUTPUT_KEY in data and "quality_2d" in data:
                skipped += 1
                continue
        stem = npz_path.stem
        target_name = stem.split("_", 1)[-1] if "_" in stem else stem
        if target_name in img_index:
            matched_pairs.append((img_index[target_name], npz_path))

    print(f"📦 匹配到 {len(matched_pairs)} 个新样本 (跳过 {skipped} 个已处理)")
    if not matched_pairs:
        return

    # 2. 推理准备
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE)
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loader = DataLoader(
        ImageBatchDataset(matched_pairs, preprocess),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # 3. 推理并回写
    with torch.no_grad():
        for batch_imgs, batch_paths, batch_qs in tqdm(loader, desc="Batch Inference"):
            valid_mask = [p != "ERROR" for p in batch_paths]
            if not any(valid_mask):
                continue

            feats = model(batch_imgs.to(DEVICE))
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
            qs = batch_qs.numpy()

            for i, p_str in enumerate(batch_paths):
                if p_str == "ERROR":
                    continue
                p = Path(p_str)
                data = dict(np.load(p, allow_pickle=True))
                data[OUTPUT_KEY] = feats[i].reshape(1, -1)
                data["quality_2d"] = np.array([qs[i][0]], dtype=np.float32)
                np.savez_compressed(p, **data)

    print("✨ 2D 特征与质量因子注入完成！")


if __name__ == "__main__":
    main()
