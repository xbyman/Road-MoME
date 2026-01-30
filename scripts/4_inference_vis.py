"""
[Step 4] MoME 批量可视化推理脚本 (块状网格修复版)
功能：
1. 块状视觉风格：使用 imshow 将 1D Patch 结果还原为规整的 2D 物理网格。
2. 筛选引擎：支持 all (全部), limit (限制数量), random (随机), specific (指定文件名) 四种模式。
3. 边界修复：修正 X 轴 extent 逻辑，确保 1m x 1m 的 Patch 能够完整覆盖到 ROI 边界（3.0m）。
4. 专家图例：第三栏清晰展示 Phys, 3D-Geom, 2D-Tex 三大专家的决策权重分布。
"""

import os
import sys
import torch
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到系统路径以导入自定义模型
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"❌ 配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
PATH_CFG = cfg["paths"]
GEO_CFG = cfg["geometry"]
FEAT_CFG = cfg["features"]
INF_CFG = cfg.get("inference", {})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================


def select_inference_files(all_files):
    """
    基于配置文件的 mode 筛选待推理的文件
    """
    mode = INF_CFG.get("mode", "limit")

    if mode == "all":
        return all_files

    elif mode == "limit":
        limit = INF_CFG.get("batch_limit", 10)
        return all_files[:limit] if limit > 0 else all_files

    elif mode == "random":
        count = INF_CFG.get("random_count", 10)
        return random.sample(all_files, min(count, len(all_files)))

    elif mode == "specific":
        targets = INF_CFG.get("select_files", [])
        selected = [f for f in all_files if any(t in f.name for t in targets)]
        return selected

    return all_files[:10]


def run_batch_inference():
    # 1. 初始化并加载训练好的模型权重
    model_path = PATH_CFG["weights"]["mome_model"]
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型权重文件: {model_path}")
        return

    model = build_mome_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✅ 成功加载权重: {Path(model_path).name} | 执行模式: {INF_CFG.get('mode')}")

    # 2. 筛选特征包
    npz_dir = Path(PATH_CFG["output_dir"])
    all_npz = sorted(list(npz_dir.glob("*.npz")))
    npz_files = select_inference_files(all_npz)

    if not npz_files:
        print(f"⚠️ 未发现可处理的特征包，请检查模式设置或 output_dir。")
        return

    # 3. 准备输出目录
    vis_dir = Path(
        PATH_CFG.get("vis_output_dir", project_root / "data" / "inference_vis")
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 4. 计算网格参数 (用于 imshow 的矩阵还原)
    # 根据 Step 0 的切片逻辑计算网格
    step = GEO_CFG["patch_size"] * (1 - GEO_CFG["overlap"])
    x_bins = np.arange(
        GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    y_bins = np.arange(
        GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    num_x, num_y = len(x_bins), len(y_bins)

    # 修复 X 轴不到 3 的关键逻辑：设置绘图的物理边界范围
    x_extent = [x_bins[0], x_bins[-1] + GEO_CFG["patch_size"]]
    y_extent = [y_bins[0], y_bins[-1] + GEO_CFG["patch_size"]]

    print(f"🚀 开始生成块状热力图，共计 {len(npz_files)} 帧...")

    # 5. 循环处理
    for npz_path in tqdm(npz_files, desc="Processing"):
        try:
            # A. 加载特征
            data = np.load(npz_path, allow_pickle=True)
            k3d, k2d = FEAT_CFG["3d"]["key_name"], FEAT_CFG["2d"]["key_name"]

            phys = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(DEVICE)
            geom = torch.from_numpy(data[k3d]).float().unsqueeze(0).to(DEVICE)
            tex = torch.from_numpy(data[k2d]).float().unsqueeze(0).to(DEVICE)
            labels = data["meta"][:, 0]

            # B. 模型推理
            with torch.no_grad():
                logits, weights = model(phys, geom, tex)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                expert_weights = weights.squeeze(0).cpu().numpy()

            # C. 填充二维热力图矩阵 (块状结构核心)
            heatmap_pred = np.zeros((num_y, num_x))
            heatmap_gt = np.zeros((num_y, num_x))
            expert_map = np.zeros((num_y, num_x))

            idx = 0
            # 嵌套循环顺序需与 Step 0 的切片顺序 [X, Y] 保持完全一致
            for i in range(num_x):
                for j in range(num_y):
                    if idx < len(probs):
                        heatmap_pred[j, i] = probs[idx]
                        heatmap_gt[j, i] = labels[idx]
                        # 记录权重最高的专家 (0:Phys, 1:3D, 2:2D)
                        expert_map[j, i] = np.argmax(expert_weights[idx])
                        idx += 1

            # D. 绘图 (还原 imshow 网格风格)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 1. MoME 异常概率热力图
            im1 = axes[0].imshow(
                heatmap_pred,
                cmap="jet",
                extent=[*x_extent, *y_extent],
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[0].set_title("MoME Prediction (Anomaly Prob)")
            axes[0].set_xlabel("Width X (m)")
            axes[0].set_ylabel("Length Y (m)")
            fig.colorbar(im1, ax=axes[0])

            # 2. 几何伪标签参考
            im2 = axes[1].imshow(
                heatmap_gt,
                cmap="Reds",
                extent=[*x_extent, *y_extent],
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1].set_title("Pseudo-Label (GT)")
            axes[1].set_xlabel("Width X (m)")
            fig.colorbar(im2, ax=axes[1])

            # 3. 主导专家分布图
            im3 = axes[2].imshow(
                expert_map,
                cmap="viridis",
                extent=[*x_extent, *y_extent],
                origin="lower",
                vmin=0,
                vmax=2,
            )
            axes[2].set_title("Dominant Expert (0:P, 1:G, 2:T)")
            axes[2].set_xlabel("Width X (m)")
            cbar3 = fig.colorbar(im3, ax=axes[2], ticks=[0, 1, 2])
            cbar3.ax.set_yticklabels(["Phys", "3D-Geom", "2D-Tex"])

            plt.suptitle(f"MoME Frame Analysis: {npz_path.name}", fontsize=14)
            plt.tight_layout()

            # E. 保存结果
            save_path = vis_dir / f"vis_{npz_path.stem}.png"
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

        except Exception as e:
            print(f"❌ 处理帧 {npz_path.name} 出错: {e}")

    print(f"\n✨ 批量推理完成！块状报告已存入: {vis_dir}")


if __name__ == "__main__":
    run_batch_inference()
