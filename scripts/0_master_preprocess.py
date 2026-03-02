"""
[Step 0-Final] 弱监督集成预处理脚本 (稳定复刻版)
功能：
1. 采用验证成功的 12 字节二进制解析逻辑。
2. 继承稳定版的 RANSAC 扶平与旋转中心逻辑。
3. 补全 phys_8d 物理特征计算 (不再硬编码 0)。
4. 引入局部归零 (np.median) 以确保即便 ROI 扩大，max_dz 依然稳定。
"""

import os
import numpy as np
import open3d as o3d
import torch
import yaml
import warnings
from tqdm import tqdm
from pathlib import Path

# 忽略 Open3D 冗余警告
warnings.filterwarnings("ignore")

import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ==================== 配置加载 ====================
def load_config():
    cfg_path = os.path.join(project_root, "config", "config.yaml")
    if not os.path.exists(cfg_path):
        # 如果找不到配置，使用你当前的复现参数
        return {
            "paths": {
                "raw_pcd_dir": r"E:\road_segmentation",
                "output_dir": r"C:\Users\31078\Desktop\ROAD\data\frame_packages",
            },
            "geometry": {
                "roi_x": [-3.0, 3.0],
                "roi_y": [-8.0, 0],
                "patch_size": 1.0,
                "overlap": 0.2,
                "th_anomaly": 0.035,
                "sigma_q": 0.015,
            },
            "sampling": {"n_points": 8192, "k_z_stretch": 10.0},
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
PATH_CFG = cfg["paths"]
GEO_CFG = cfg["geometry"]
SAMP_CFG = cfg["sampling"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================


def read_pcd_12byte(path):
    """[验证成功] 12 字节安全读取"""
    try:
        with open(path, "rb") as f:
            data = f.read()
            pos = data.find(b"DATA binary\n")
            if pos == -1:
                return None
            raw_data = data[pos + 12 :]
            n_points = len(raw_data) // 12
            dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
            arr = np.frombuffer(raw_data[: n_points * 12], dtype=dt)
            pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
            pts = pts[np.isfinite(pts).all(axis=1)]
            return pts[np.abs(pts).max(axis=1) < 1000.0]
    except:
        return None


def fps_weighted_gpu(points, npoint, k_z=10.0):
    """[验证成功] GPU 加权 FPS 采样"""
    if len(points) < npoint:
        idxs = np.random.choice(len(points), npoint, replace=True)
        return points[idxs]
    xyz = torch.from_numpy(points).float().to(DEVICE).unsqueeze(0)
    xyz_weighted = xyz.clone()
    xyz_weighted[:, :, 2] *= k_z
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=DEVICE)
    distance = torch.ones(B, N, device=DEVICE) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=DEVICE)
    batch_indices = torch.arange(B, dtype=torch.long, device=DEVICE)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_weighted[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz_weighted - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return xyz[0, centroids[0]].cpu().numpy()


def extract_8d_and_label(pts, center_xy):
    """提取物理特征并生成弱监督标签"""
    z = pts[:, 2]
    # 使用 95 分位数对抗可能存在的离群飞点
    max_dz = np.percentile(np.abs(z), 95)

    # 标签与质量分
    label = 1 if max_dz > GEO_CFG["th_anomaly"] else 0
    diff = np.abs(max_dz - GEO_CFG["th_anomaly"])
    quality = np.clip(
        1.0 - np.exp(-(diff**2) / (2 * (GEO_CFG["sigma_q"] ** 2))), 0.1, 1.0
    )

    # 物理统计明细
    anomaly_mask = np.abs(z) > GEO_CFG["th_anomaly"]
    anomaly_pts = pts[anomaly_mask]
    ratio = np.mean(anomaly_mask)

    c_offset = np.zeros(2, dtype=np.float32)
    linear = 0.0
    if len(anomaly_pts) > 5:
        c_offset = np.mean(anomaly_pts[:, :2], axis=0) - center_xy
        try:
            cov = np.cov(anomaly_pts[:, :2].T)
            evals = np.linalg.eigvals(cov)
            linear = np.max(evals) / (np.sum(evals) + 1e-6)
        except:
            pass

    v_8d = np.array(
        [
            max_dz,
            ratio,
            c_offset[0],
            c_offset[1],
            linear,
            np.var(z),
            np.std(z),
            len(pts),
        ],
        dtype=np.float32,
    )

    return v_8d, label, quality


def process_frame(pcd_path):
    pts_raw = read_pcd_12byte(pcd_path)
    if pts_raw is None:
        return None

    # 1. RANSAC 全局扶平 (复刻你的稳定版逻辑)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_raw)
    try:
        plane, inliers = pcd.segment_plane(0.04, 5, 1000)
        normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        axis = np.cross(normal, [0, 0, 1])
        angle = np.arccos(np.clip(normal[2], -1.0, 1.0))
        if np.linalg.norm(axis) > 1e-6:
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis / np.linalg.norm(axis) * angle
            )
            pcd.rotate(rot, center=np.mean(pts_raw[inliers], axis=0))
        pts = np.asarray(pcd.points)
        pts[:, 2] -= np.mean(pts[inliers, 2])
    except:
        pts = pts_raw

    # 2. Patch 切分
    step = GEO_CFG["patch_size"] * (1 - GEO_CFG["overlap"])
    x_bins = np.arange(
        GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    y_bins = np.arange(
        GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - GEO_CFG["patch_size"] + 0.1, step
    )

    d_phys, d_pts, d_meta = [], [], []
    for xi in x_bins:
        for yi in y_bins:
            mask = (
                (pts[:, 0] >= xi)
                & (pts[:, 0] < xi + GEO_CFG["patch_size"])
                & (pts[:, 1] >= yi)
                & (pts[:, 1] < yi + GEO_CFG["patch_size"])
            )
            p_pts = pts[mask].copy()
            if len(p_pts) < 100:
                continue

            # 【定海神针】局部归零，消除任何残余坡度
            p_pts[:, 2] -= np.median(p_pts[:, 2])

            # 提取 8D 与打标
            v_8d, label, q = extract_8d_and_label(p_pts, np.array([xi + 0.5, yi + 0.5]))

            # 采样与归一化
            p_sampled = fps_weighted_gpu(
                p_pts, SAMP_CFG["n_points"], k_z=SAMP_CFG["k_z_stretch"]
            )
            p_sampled -= np.mean(p_sampled, axis=0)
            norm = np.max(np.linalg.norm(p_sampled, axis=1))
            if norm > 1e-7:
                p_sampled /= norm

            d_phys.append(v_8d)
            d_pts.append(p_sampled.astype(np.float32))
            d_meta.append([label, q])

    if not d_phys:
        return None
    return {
        "phys_8d": np.array(d_phys),
        "sampled_pts": np.array(d_pts),
        "meta": np.array(d_meta),
    }


def main():
    os.makedirs(PATH_CFG["output_dir"], exist_ok=True)

    # --- 核心修改：递归扫描所有子目录下的 .pcd ---
    raw_root = Path(PATH_CFG["raw_pcd_dir"])
    print(f"🔍 正在递归扫描原始目录: {raw_root}")

    # 使用 rglob 搜索所有层级下的 pcd 文件
    pcd_files = sorted(list(raw_root.rglob("*.pcd")))

    if not pcd_files:
        print("❌ 未发现任何 PCD 文件，请检查 config 中的 raw_pcd_dir。")
        return

    print(f"🚀 启动全量预处理 | 总计: {len(pcd_files)} 帧")

    for p_path in tqdm(pcd_files):
        # 为了防止不同时间戳下文件名冲突，我们在输出名中包含父目录名
        # 例如: 2023-03-17_0001.npz
        timestamp_folder = p_path.parent.parent.name  # 假设结构是 timestamp/pcd/*.pcd
        out_name = f"{timestamp_folder}_{p_path.stem}.npz"
        out_path = os.path.join(PATH_CFG["output_dir"], out_name)

        if os.path.exists(out_path):
            continue

        res = process_frame(str(p_path))

        if res is not None:
            np.savez_compressed(out_path, **res)


if __name__ == "__main__":
    main()
