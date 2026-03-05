"""
[Step 0] 几何预处理与容器化 (v3.7 增量对齐版)
功能：
1. 12字节 PCD 二进制解析。
2. 密度感知质量映射：即便点云稀疏也保留 0.15 的基础权重，防止 2D 梯度丢失。
3. 空间对齐保障：强制输出固定长度序列 (max_patches)，修复热力图“左偏”。
4. 增量处理：检测 output_dir，若 .npz 已存在则自动跳过。
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
PATH_CFG = cfg["paths"]
GEO_CFG = cfg["geometry"]
SAMP_CFG = cfg.get("sampling", {"n_points": 8192, "k_z_stretch": 10.0})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_pcd_12byte(path):
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
            return pts[np.isfinite(pts).all(axis=1)]
    except:
        return None


def fps_weighted_gpu(points, npoint, k_z=10.0):
    if len(points) == 0:
        return np.zeros((npoint, 3), dtype=np.float32)
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


def extract_patch_meta(pts, center_xy):
    n_pts = len(pts)
    if n_pts == 0:
        return np.zeros(8, dtype=np.float32), 0.0, 0.0
    z = pts[:, 2]
    max_dz = np.percentile(np.abs(z), 95)
    label = 1 if max_dz > GEO_CFG["th_anomaly"] else 0
    diff = np.abs(max_dz - GEO_CFG["th_anomaly"])
    base_q = 1.0 - np.exp(-(diff**2) / (2 * (GEO_CFG["sigma_q"] ** 2)))
    density_factor = np.clip(n_pts / 200.0, 0.1, 1.0)
    final_quality = np.clip(base_q * density_factor, 0.15, 1.0)

    v_8d = np.array(
        [
            max_dz,
            np.mean(np.abs(z) > GEO_CFG["th_anomaly"]),
            0.0,
            0.0,
            0.0,
            np.var(z),
            np.std(z),
            n_pts,
        ],
        dtype=np.float32,
    )
    return v_8d, label, final_quality


def process_frame(pcd_path):
    pts_raw = read_pcd_12byte(pcd_path)
    if pts_raw is None or len(pts_raw) < 5:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_raw)
    try:
        plane, inliers = pcd.segment_plane(0.05, 3, 500)
        pts = np.asarray(pcd.points)
        pts[:, 2] -= np.mean(pts[inliers, 2])
    except:
        pts = pts_raw

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
            v_8d, label, q = extract_patch_meta(p_pts, np.array([xi + 0.5, yi + 0.5]))
            p_sampled = fps_weighted_gpu(
                p_pts, SAMP_CFG["n_points"], k_z=SAMP_CFG["k_z_stretch"]
            )
            if len(p_pts) > 5:
                p_sampled -= np.mean(p_sampled, axis=0)
                norm = np.max(np.linalg.norm(p_sampled, axis=1))
                if norm > 1e-7:
                    p_sampled /= norm
            d_phys.append(v_8d)
            d_pts.append(p_sampled.astype(np.float32))
            d_meta.append([label, q])
    return {
        "phys_8d": np.array(d_phys),
        "sampled_pts": np.array(d_pts),
        "meta": np.array(d_meta),
    }


def main():
    os.makedirs(PATH_CFG["output_dir"], exist_ok=True)
    raw_root = Path(PATH_CFG["raw_pcd_dir"])
    pcd_files = sorted(list(raw_root.rglob("*.pcd")))
    print(f"🚀 启动增量预处理 | 扫描到 {len(pcd_files)} 帧")
    for p_path in tqdm(pcd_files):
        out_path = Path(PATH_CFG["output_dir"]) / f"pkg_{p_path.stem}.npz"
        if out_path.exists():
            continue
        res = process_frame(p_path)
        if res:
            np.savez_compressed(out_path, **res)
    print("✨ 预处理任务结束。")


if __name__ == "__main__":
    main()
