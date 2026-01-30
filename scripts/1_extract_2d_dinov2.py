"""
[Step 1] DINOv2 视觉特征提取与注入脚本 (高性能版)
优化点：
1. 预索引：启动时全量扫描图像目录并建立缓存，消除循环内磁盘 IO 瓶颈。
2. 批处理：利用 DataLoader 多线程读图 + GPU Batch 推理，极大提升吞吐量。
3. [新增] 断点续传：自动检查 NPZ 内部是否已含 2D 特征，若存在则跳过，节省重复推理时间。
"""

import os
import sys
import torch
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"❌ 配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
PRE_CFG = config.get("preprocess", {})
PATH_CFG = config["paths"]
FEAT_CFG = config["features"]["2d"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = PRE_CFG.get("inference_batch_size", 32)
NUM_WORKERS = PRE_CFG.get("num_workers", 4)
NPZ_DIR = Path(PATH_CFG["output_dir"])
IMG_DIR = Path(PATH_CFG["raw_img_dir"])
OUTPUT_KEY = FEAT_CFG["key_name"]
# ===============================================


class ImageBatchDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, npz_path = self.pairs[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, str(npz_path)
        except Exception:
            return torch.zeros(3, 518, 518), "ERROR"


def main():
    print("=" * 60)
    print(f"🚀 启动 2D 特征加速提取流水线 (DINOv2)")
    print(f"⚙️  配置: 批大小={BATCH_SIZE}, 工作线程={NUM_WORKERS}, 设备={DEVICE}")
    print("=" * 60)

    # 1. 建立图像路径预索引
    print(f"🔍 正在扫描图像目录并建立索引: {IMG_DIR}...")
    img_index = {}
    valid_exts = {".jpg", ".png", ".jpeg"}
    for p in IMG_DIR.rglob("*"):
        if p.suffix.lower() in valid_exts:
            img_index[p.stem] = p
    print(f"✅ 索引建立完成，共发现 {len(img_index)} 张图像。")

    # 2. 匹配 NPZ 与图像，并增加断点检查
    npz_files = sorted(list(NPZ_DIR.glob("*.npz")))
    matched_pairs = []
    skipped_count = 0

    print(f"📂 正在检查特征包对齐情况与提取进度...")
    for npz_path in tqdm(npz_files, desc="Checking progress"):
        try:
            # --- 断点续传逻辑 ---
            # 加载并检查是否已有特征键名
            with np.load(npz_path) as data:
                if OUTPUT_KEY in data:
                    skipped_count += 1
                    continue

            stem = npz_path.stem
            target_name = stem.split("_", 1)[-1] if "_" in stem else stem

            if target_name in img_index:
                matched_pairs.append((img_index[target_name], npz_path))
            elif stem in img_index:
                matched_pairs.append((img_index[stem], npz_path))
        except Exception:
            continue

    if skipped_count > 0:
        print(f"ℹ️  跳过了 {skipped_count} 个已包含 2D 特征的 NPZ 文件。")

    if not matched_pairs:
        if skipped_count > 0:
            print("✅ 任务已全部完成，无需重新提取。")
        else:
            print("⚠️ 未发现可匹配的数据对，请检查文件名对应关系。")
        return

    print(f"📦 成功匹配 {len(matched_pairs)} 个待提取样本，准备推理...")

    # 3. 加载模型与预处理
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

    # 4. 启动 DataLoader
    dataset = ImageBatchDataset(matched_pairs, preprocess)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    success_count = 0
    last_feat_shape = "None"

    # 5. 批处理推理与存盘
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(loader, desc="推理进度"):
            valid_mask = [p != "ERROR" for p in batch_paths]
            if not any(valid_mask):
                continue

            imgs = batch_imgs.to(DEVICE)
            features = model(imgs)
            features = torch.nn.functional.normalize(features, dim=-1)
            feats_np = features.cpu().numpy()

            for i, npz_path_str in enumerate(batch_paths):
                if npz_path_str == "ERROR":
                    continue
                try:
                    p = Path(npz_path_str)
                    data = dict(np.load(p, allow_pickle=True))

                    feat_single = feats_np[i].reshape(1, -1)
                    data[OUTPUT_KEY] = feat_single
                    data["quality_2d"] = np.array([1.0], dtype=np.float32)

                    np.savez_compressed(p, **data)
                    success_count += 1
                    last_feat_shape = str(feat_single.shape)
                except Exception as e:
                    print(f"❌ 写回 {npz_path_str} 失败: {e}")

    print("\n" + "=" * 60)
    print(f"✨ 2D 特征提取增量更新完成！")
    print(f"   本次更新: {success_count} 帧 | 注入维度: {last_feat_shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
