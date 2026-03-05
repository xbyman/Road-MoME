"""
[Step 4] MoME 深度诊断可视化 (v2.8 终极学术版)
功能：针对 v2.8 架构进行全维度诊断，支持指定样本分析。
升级：
1. 2x4 矩阵布局：新增“3D专家权重”与“专家权重漂移统计柱状图”。
2. 坐标轴修复：彻底解决物理坐标映射的 Extent 偏移问题。
3. 质量感知显示：实时标注该帧的图像质量分 q2。
4. 鲁棒性支持：适配 v2.8 的四张量输出 (logits, weights, exp_logits, uncerts)。
"""

import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import random

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
GEO_CFG = cfg["geometry"]
PATH_CFG = cfg["paths"]
INF_CFG = cfg["inference"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruct_grid(values, x_bins, y_bins):
    """将一维 Patch 序列还原为物理空间二维网格"""
    grid = np.zeros((len(y_bins), len(x_bins)))
    idx = 0
    # 遵循 preprocess 时的嵌套顺序：先 X 后 Y
    for i in range(len(x_bins)):
        for j in range(len(y_bins)):
            if idx < len(values):
                grid[j, i] = values[idx]
            idx += 1
    return grid


def run_diagnostic():
    # 1. 初始化模型
    model = build_mome_model(cfg).to(DEVICE)
    model_path = PATH_CFG["weights"]["mome_model"]
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重: {model_path}，请先运行训练脚本。")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 准备文件列表
    all_npz = sorted(list(Path(PATH_CFG["output_dir"]).glob("*.npz")))
    if INF_CFG["mode"] == "select":
        target_ids = INF_CFG["select_files"]
        npz_files = [p for p in all_npz if any(tid in p.name for tid in target_ids)]
    elif INF_CFG["mode"] == "random":
        npz_files = random.sample(all_npz, min(len(all_npz), INF_CFG["batch_limit"]))
    else:
        npz_files = all_npz[: INF_CFG["batch_limit"]]

    if not npz_files:
        print("⚠️ 未找到有效样本，请检查 config 中的 select_files 列表。")
        return

    # 物理参数计算
    step = GEO_CFG["patch_size"] * (1 - GEO_CFG["overlap"])
    x_bins = np.arange(
        GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    y_bins = np.arange(
        GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - GEO_CFG["patch_size"] + 0.1, step
    )

    # 修正后的 Extent 范围
    extent = [
        GEO_CFG["roi_x"][0],
        GEO_CFG["roi_x"][1],
        GEO_CFG["roi_y"][0],
        GEO_CFG["roi_y"][1],
    ]

    vis_dir = Path(PATH_CFG["vis_output_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 启动 v2.8 全景诊断 | 样本总数: {len(npz_files)}")

    for p in tqdm(npz_files, desc="Diagnostic Inference"):
        try:
            data = np.load(p, allow_pickle=True)
            phys = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(DEVICE)
            geom = torch.from_numpy(data["deep_512d"]).float().unsqueeze(0).to(DEVICE)
            tex = torch.from_numpy(data["deep_2d_768d"]).float().unsqueeze(0).to(DEVICE)
            meta = data["meta"]  # [N, 2] -> [Label, Geo_Quality]

            # 质量因子与输入构造
            q2_val = data.get("quality_2d", np.array([INF_CFG.get("q2_default", 0.8)]))[
                0
            ]
            q_geo = torch.from_numpy(meta[:, 1:2]).float()
            q_vec = (
                torch.cat([q_geo, torch.full_like(q_geo, q2_val)], dim=-1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                # [适配 v2.8]
                logits, weights, _, uncerts = model(phys, geom, tex, q_vec)

                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                w_all = weights.squeeze(0).cpu().numpy()  # [N, 4]
                dominant_idx = np.argmax(w_all, axis=1)
                # 视觉不确定性: sigma = exp(0.5 * s)
                sigma_tex = torch.exp(0.5 * uncerts[0, :, 1]).cpu().numpy()

            # --- 绘图逻辑 (2x4 矩阵) ---
            fig, axes = plt.subplots(2, 4, figsize=(24, 11), dpi=100)
            fig.suptitle(
                f"MI-MoE v2.8 Full Holistic Diagnostic | Sample: {p.stem}\nVisual Quality (q2): {q2_val:.3f}",
                fontsize=18,
                fontweight="bold",
            )

            # [0, 0] 预测图
            im1 = axes[0, 0].imshow(
                reconstruct_grid(probs, x_bins, y_bins),
                extent=extent,
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[0, 0].set_title("1. Detection Prob (Fusion)", fontsize=12)
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # [0, 1] 伪标签
            im2 = axes[0, 1].imshow(
                reconstruct_grid(meta[:, 0], x_bins, y_bins),
                extent=extent,
                cmap="Reds",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[0, 1].set_title("2. Pseudo-Label (Baseline)", fontsize=12)
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # [0, 2] 主导专家分布
            expert_cmap = ListedColormap(
                ["#bdc3c7", "#3498db", "#f1c40f", "#ffffff"]
            )  # Phys, 3D, 2D, Syn
            im3 = axes[0, 2].imshow(
                reconstruct_grid(dominant_idx, x_bins, y_bins),
                extent=extent,
                cmap=expert_cmap,
                origin="lower",
                vmin=0,
                vmax=3,
            )
            axes[0, 2].set_title("3. Dominant Expert Distribution", fontsize=12)
            cbar3 = plt.colorbar(
                im3, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3]
            )
            cbar3.ax.set_yticklabels(["Phys", "3D-Geom", "2D-Tex", "Synergy"])

            # [0, 3] 专家权重漂移统计 (新增)
            avg_weights = w_all.mean(axis=0)
            labels = ["Phys", "3D", "2D", "Syn"]
            colors = ["#bdc3c7", "#3498db", "#f1c40f", "#9b59b6"]
            axes[0, 3].bar(labels, avg_weights, color=colors, alpha=0.8)
            axes[0, 3].set_ylim(0, 1.0)
            axes[0, 3].set_title("4. Weight Distribution Analysis", fontsize=12)
            for i, v in enumerate(avg_weights):
                axes[0, 3].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

            # [1, 0] 协同权重 (Synergy)
            im4 = axes[1, 0].imshow(
                reconstruct_grid(w_all[:, 3], x_bins, y_bins),
                extent=extent,
                cmap="Greens",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 0].set_title("5. Synergy Weight (w_syn)", fontsize=12)
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # [1, 1] 几何专家权重 (3D) - (新增)
            im5 = axes[1, 1].imshow(
                reconstruct_grid(w_all[:, 1], x_bins, y_bins),
                extent=extent,
                cmap="Blues",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 1].set_title("6. 3D-Geom Weight (w_3d)", fontsize=12)
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # [1, 2] 纹理专家权重 (2D)
            im6 = axes[1, 2].imshow(
                reconstruct_grid(w_all[:, 2], x_bins, y_bins),
                extent=extent,
                cmap="YlOrBr",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 2].set_title("7. Texture Weight (w_2d)", fontsize=12)
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

            # [1, 3] 视觉不确定性 (Sigma)
            im7 = axes[1, 3].imshow(
                reconstruct_grid(sigma_tex, x_bins, y_bins),
                extent=extent,
                cmap="magma",
                origin="lower",
            )
            axes[1, 3].set_title("8. Visual Uncertainty (Sigma)", fontsize=12)
            plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

            # 统一坐标标签
            for i in range(2):
                for j in range(4):
                    if not (i == 0 and j == 3):  # 柱状图不加坐标标签
                        axes[i, j].set_xlabel("X (m)")
                        axes[i, j].set_ylabel("Y (m)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(vis_dir / f"full_diagnostic_v28_{p.stem}.png", dpi=120)
            plt.close()

        except Exception as e:
            print(f"❌ 帧 {p.name} 诊断处理失败: {str(e)}")

    print(f"✨ 诊断报告已全部生成至: {vis_dir}")


if __name__ == "__main__":
    run_diagnostic()
