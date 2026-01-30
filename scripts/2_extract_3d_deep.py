"""
[Step 2] Point-MAE 几何深度特征提取脚本 (显存保护与断点续传版)
优化点：
1. 提取检查：支持从 config 读取 skip_existing，自动跳过已处理的文件。
2. 显存清理：增加显存主动回收，适配 8GB 显存环境。
3. 磁盘同步：每处理 500 个 Patch 自动保存一次，防止内存积压和意外中断。
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.backbones import RoadPointMAEEncoder, load_official_pretrain


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"❌ 配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
PATH_CFG = cfg["paths"]
FEAT_CFG = cfg["features"]["3d"]
SAMP_CFG = cfg["sampling"]
PRE_CFG = cfg.get("preprocess", {})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = PRE_CFG.get("inference_batch_size", 8)
SKIP_EXISTING = PRE_CFG.get("skip_existing", True)
NPZ_DIR = Path(PATH_CFG["output_dir"])
CKPT_PATH = PATH_CFG["weights"]["point_mae"]
OUTPUT_KEY = FEAT_CFG["key_name"]
TRANS_DIM = SAMP_CFG.get("trans_dim", 384)
# ===============================================


class PatchFeatureDataset(Dataset):
    """
    数据集加载器：从 .npz 中提取待推理的 sampled_pts
    """

    def __init__(self, npz_files, output_key, skip_flag):
        self.samples = []
        self.skipped_count = 0
        print(f"🔍 正在扫描特征包并检查 3D 提取进度...")

        for f in tqdm(npz_files, desc="扫描中"):
            try:
                # 检查是否开启了跳过已处理逻辑
                if skip_flag:
                    with np.load(f, allow_pickle=True) as data:
                        if output_key in data:
                            self.skipped_count += 1
                            continue

                        if "sampled_pts" in data:
                            pts_array = data["sampled_pts"]
                            for i in range(len(pts_array)):
                                self.samples.append((str(f), i, pts_array[i]))
                else:
                    data = np.load(f, allow_pickle=True)
                    if "sampled_pts" in data:
                        pts_array = data["sampled_pts"]
                        for i in range(len(pts_array)):
                            self.samples.append((str(f), i, pts_array[i]))
            except Exception:
                continue

        if self.skipped_count > 0:
            print(
                f"ℹ️  检测到已完成任务，自动跳过 {self.skipped_count} 个已提取 3D 特征的 NPZ。"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, patch_idx, pts = self.samples[idx]
        return torch.from_numpy(pts).float(), npz_path, patch_idx


def main():
    print("=" * 60)
    print(f"🚀 启动 3D 深度特征提取流水线 (断点续传模式)")
    print(f"⚙️  配置: 批大小={BATCH_SIZE}, 跳过已处理={SKIP_EXISTING}, 设备={DEVICE}")
    print("=" * 60)

    # 1. 初始化并加载模型
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到 3D 模型权重: {CKPT_PATH}")
        return

    model = RoadPointMAEEncoder(trans_dim=TRANS_DIM).to(DEVICE)
    model = load_official_pretrain(model, CKPT_PATH)
    model.eval()

    # 2. 准备数据
    npz_files = sorted(list(NPZ_DIR.glob("*.npz")))
    if not npz_files:
        print(f"⚠️  在 {NPZ_DIR} 中未发现特征包，请先运行 Step 0。")
        return

    dataset = PatchFeatureDataset(npz_files, OUTPUT_KEY, SKIP_EXISTING)

    if len(dataset) == 0:
        print("✅ 检查完毕：所有样本的 3D 特征均已存在，无需重复提取。")
        return

    # num_workers=0 在 8GB 显卡环境下最稳
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 批处理推理
    print(f"🧪 开始特征提取，总计 {len(dataset)} 个 Patch 待处理...")

    results_cache = {}  # {npz_path: {idx: feat}}
    write_freq = 500  # 内存阈值：每积累 500 个 Patch 结果就强制写入磁盘
    processed_count = 0

    with torch.no_grad():
        for batch_pts, batch_paths, batch_idxs in tqdm(loader, desc="推理进度"):
            # 准备输入
            B_curr = batch_pts.shape[0]
            center = batch_pts[:, :128, :].to(DEVICE)
            neighborhood_grouped = (
                batch_pts[:, :8192, :].view(B_curr, 128, 64, 3).to(DEVICE)
            )

            # 推理
            feats = model(neighborhood_grouped, center)
            feats_np = feats.cpu().numpy()

            # 存入缓存
            for i in range(len(batch_paths)):
                path = batch_paths[i]
                if path not in results_cache:
                    results_cache[path] = {}
                results_cache[path][int(batch_idxs[i])] = feats_np[i]

            processed_count += B_curr

            # 定期清理显存与同步磁盘，防止死机
            if processed_count % (BATCH_SIZE * 20) == 0:
                torch.cuda.empty_cache()

            if len(results_cache) >= write_freq:
                save_cache_to_npz(results_cache, TRANS_DIM, OUTPUT_KEY)
                results_cache.clear()

    # 最后剩余结果写回
    if results_cache:
        save_cache_to_npz(results_cache, TRANS_DIM, OUTPUT_KEY)

    print("\n" + "=" * 60)
    print(f"✨ 3D 特征提取任务圆满完成！")
    print("=" * 60)


def save_cache_to_npz(cache, dim, key):
    """磁盘同步辅助函数"""
    for npz_path_str, patch_data in cache.items():
        try:
            p = Path(npz_path_str)
            data = dict(np.load(p, allow_pickle=True))
            num_patches = len(data["phys_8d"])

            sorted_feats = []
            for i in range(num_patches):
                sorted_feats.append(patch_data.get(i, np.zeros(dim)))

            data[key] = np.array(sorted_feats, dtype=np.float32)
            np.savez_compressed(p, **data)
        except Exception as e:
            print(f"❌ 写入 {npz_path_str} 失败: {e}")


if __name__ == "__main__":
    main()
