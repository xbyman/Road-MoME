"""
[Step 0] 流式预处理与打包脚本 (Master Preprocess - 对齐增强修复版)
功能：
1. 读取原始 PCD (修正为 12 字节 x,y,z 步长)。
2. 增加数值清洗，防止 Inf/NaN 数值爆炸。
3. 执行 RANSAC 扶平，即使失败也保留原始数据以确保 1:1 对齐。
4. 提取 8 维物理特征 + 质量权重。
5. 执行加权 FPS 采样 (Z轴增强)，生成 8192 点标准化点云。
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

# 添加项目根目录到路径
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ==================== 配置加载 ====================
CONFIG_PATH = os.path.join(project_root, "config", "config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
        PATH_CFG = full_config["paths"]
        GEO_CFG = full_config["geometry"]
        SAMP_CFG = full_config["sampling"]
else:
    # 默认兜底配置
    PATH_CFG = {
        "raw_pcd_dir": r"E:\road_segmentation",
        "output_dir": r"C:\Users\31078\Desktop\Road_Project\data\frame_packages",
    }
    GEO_CFG = {
        "roi_x": [-5.0, 5.0],
        "roi_y": [-5.0, 5.0],
        "patch_size": 1.0,
        "overlap": 0.2,
        "th_anomaly": 0.035,
        "sigma_q": 0.015,
    }
    SAMP_CFG = {"n_points": 8192, "k_z_stretch": 10.0}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================


def read_pcd_custom(path):
    """
    [关键修复] 针对 FIELDS x y z (12字节) 的安全读取
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
            header_end = b"DATA binary\n"
            pos = data.find(header_end)
            if pos == -1:
                return None

            raw_data = data[pos + len(header_end) :]
            # 修正：12 字节 = 4(x) + 4(y) + 4(z)
            n_points = len(raw_data) // 12
            clean_data = raw_data[: n_points * 12]

            dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
            arr = np.frombuffer(clean_data, dtype=dt)
            pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)

            # --- 数值清洗：剔除离群噪点 ---
            # 过滤 NaN/Inf
            pts = pts[np.isfinite(pts).all(axis=1)]
            # 过滤物理意义上的离谱点 (绝对值大于1000m)
            mask_safe = np.abs(pts).max(axis=1) < 1000.0
            pts = pts[mask_safe]

            return pts
    except Exception as e:
        print(f"❌ 读取错误 {path}: {e}")
        return None


def fps_weighted_gpu(points, npoint, k_z=10.0):
    """加权最远点采样"""
    if len(points) < npoint:
        idxs = np.random.choice(len(points), npoint, replace=True)
        return points[idxs]

    xyz = torch.from_numpy(points).float().to(DEVICE).unsqueeze(0)
    B, N, C = xyz.shape
    xyz_weighted = xyz.clone()
    xyz_weighted[:, :, 2] *= k_z  # 增强 Z 轴敏感度

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

    idx = centroids[0].cpu().numpy()
    return points[idx]


def extract_8d_stats(pts, center):
    """提取 8 维鲁棒物理特征"""
    if len(pts) < 10:
        return np.zeros(8, dtype=np.float32)

    z = pts[:, 2]
    # 使用有限值计算，防止 Inf 干扰
    max_dz = np.max(np.abs(z))

    anomaly_mask = np.abs(z) > GEO_CFG["th_anomaly"]
    anomaly_pts = pts[anomaly_mask]
    ratio = len(anomaly_pts) / len(pts)

    c_offset = (
        np.mean(anomaly_pts[:, :2], axis=0) - center
        if len(anomaly_pts) > 0
        else np.zeros(2)
    )

    linear = 0.0
    if len(anomaly_pts) > 5:
        try:
            cov = np.cov(anomaly_pts[:, :2].T)
            evals = np.linalg.eigvals(cov)
            linear = np.max(evals) / (np.sum(evals) + 1e-6)
        except:
            pass

    return np.array(
        [
            max_dz,
            ratio,
            c_offset[0],
            c_offset[1],
            linear,
            np.var(z),
            np.std(z),
            len(pts) / (GEO_CFG["patch_size"] ** 2),
        ],
        dtype=np.float32,
    )


def process_frame(pcd_path):
    """处理单帧核心逻辑"""
    xyz = read_pcd_custom(pcd_path)
    if xyz is None:
        return None

    # 1. 扶平校正
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pts = xyz

    try:
        # 即使内点较少也尝试拟合，失败也不 return None 以保证 1:1 匹配
        plane, inliers = pcd.segment_plane(0.04, 3, 500)
        if len(inliers) > 100:
            normal = np.array(plane[:3]) / (np.linalg.norm(plane[:3]) + 1e-8)
            axis = np.cross(normal, [0, 0, 1])
            angle = np.arccos(np.clip(normal[2], -1.0, 1.0))
            if np.linalg.norm(axis) > 1e-6:
                rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    axis / np.linalg.norm(axis) * angle
                )
                pcd.rotate(rot, center=np.mean(xyz[inliers], axis=0))

            pts = np.asarray(pcd.points)
            # 高度归零
            pts[:, 2] -= np.mean(pts[inliers, 2])
    except:
        pass  # 拟合失败直接使用原始点云

    # 2. Patch 切分
    step = GEO_CFG["patch_size"] * (1 - GEO_CFG["overlap"])
    x_bins = np.arange(
        GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    y_bins = np.arange(
        GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - GEO_CFG["patch_size"] + 0.1, step
    )

    data_phys, data_pts, data_meta = [], [], []

    for xi in x_bins:
        for yi in y_bins:
            mask = (
                (pts[:, 0] >= xi)
                & (pts[:, 0] < xi + GEO_CFG["patch_size"])
                & (pts[:, 1] >= yi)
                & (pts[:, 1] < yi + GEO_CFG["patch_size"])
            )
            p_pts = pts[mask]

            if len(p_pts) < 50:
                continue  # 降门槛保留更多边缘

            # A. 物理特征
            center = np.array(
                [xi + GEO_CFG["patch_size"] / 2, yi + GEO_CFG["patch_size"] / 2]
            )
            v_8d = extract_8d_stats(p_pts, center)

            # B. 采样与归一化
            p_sampled = fps_weighted_gpu(
                p_pts, SAMP_CFG["n_points"], k_z=SAMP_CFG["k_z_stretch"]
            )
            p_sampled -= np.mean(p_sampled, axis=0)
            dist = np.max(np.sqrt(np.sum(p_sampled**2, axis=1)))
            if dist > 0:
                p_sampled /= dist

            # C. 元数据
            label = 1 if v_8d[0] > GEO_CFG["th_anomaly"] else 0
            diff = np.abs(v_8d[0] - GEO_CFG["th_anomaly"])
            quality = np.clip(
                1.0 - np.exp(-(diff**2) / (2 * (GEO_CFG["sigma_q"] ** 2))), 0.1, 1.0
            )

            data_phys.append(v_8d)
            data_pts.append(p_sampled)
            data_meta.append([label, quality])

    return {
        "phys_8d": (
            np.array(data_phys, dtype=np.float32) if data_phys else np.zeros((0, 8))
        ),
        "sampled_pts": (
            np.array(data_pts, dtype=np.float32)
            if data_pts
            else np.zeros((0, SAMP_CFG["n_points"], 3))
        ),
        "meta": (
            np.array(data_meta, dtype=np.float32) if data_meta else np.zeros((0, 2))
        ),
    }


def main():
    os.makedirs(PATH_CFG["output_dir"], exist_ok=True)
    pcd_files = sorted(
        [f for f in os.listdir(PATH_CFG["raw_pcd_dir"]) if f.endswith(".pcd")]
    )

    print(f"🚀 启动 Master Preprocess (1:1对齐模式) | 目标: {len(pcd_files)} 帧")

    for fname in tqdm(pcd_files):
        out_path = os.path.join(PATH_CFG["output_dir"], f"{Path(fname).stem}.npz")
        if os.path.exists(out_path):
            continue

        res = process_frame(os.path.join(PATH_CFG["raw_pcd_dir"], fname))

        # 强制保存，即使是空包也要占位，确保与图像对齐
        if res is not None:
            np.savez_compressed(out_path, **res)
        else:
            np.savez_compressed(
                out_path,
                phys_8d=np.zeros((0, 8)),
                sampled_pts=np.zeros((0, SAMP_CFG["n_points"], 3)),
                meta=np.zeros((0, 2)),
            )


if __name__ == "__main__":
    main()
