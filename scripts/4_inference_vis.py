"""
[Step 4] MoME 深度诊断可视化脚本 (v3.5)
核心升级：
1. 六维矩阵：展示 [Prob, GT, Dominant, 3D_Weight, 2D_Weight, Syn_Weight]。
2. 图像融合：如果存在原始 JPG，将其作为背景或参考图展示（需确保路径对齐）。
3. 证据链追踪：显式展示协同专家是如何在病害区域“接管”决策的。
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
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


# ==================== 配置加载 ====================
def load_config():
    cfg_path = project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
PATH_CFG = cfg["paths"]
GEO_CFG = cfg["geometry"]
FEAT_CFG = cfg["features"]
INF_CFG = cfg.get("inference", {})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_diagnostic_vis():
    # 1. 加载模型
    model_path = PATH_CFG["weights"]["mome_model"]
    model = build_mome_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 准备数据
    npz_dir = Path(PATH_CFG["output_dir"])
    npz_files = sorted(list(npz_dir.glob("*.npz")))

    if INF_CFG.get("mode") == "random":
        npz_files = random.sample(
            npz_files, min(INF_CFG.get("random_count", 10), len(npz_files))
        )
    else:
        npz_files = npz_files[: INF_CFG.get("batch_limit", 10)]

    vis_dir = Path(
        PATH_CFG.get("vis_output_dir", project_root / "data" / "inference_vis")
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 3. 图像对齐准备
    img_dir = Path(PATH_CFG["raw_img_dir"])
    img_index = {p.stem: p for p in img_dir.rglob("*.jpg")}

    # 4. 网格参数 (对齐 roi_y: [-8.0, 0])
    step = GEO_CFG["patch_size"] * (1 - GEO_CFG["overlap"])
    x_bins = np.arange(
        GEO_CFG["roi_x"][0], GEO_CFG["roi_x"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    y_bins = np.arange(
        GEO_CFG["roi_y"][0], GEO_CFG["roi_y"][1] - GEO_CFG["patch_size"] + 0.1, step
    )
    num_x, num_y = len(x_bins), len(y_bins)
    extent = [
        x_bins[0],
        x_bins[-1] + GEO_CFG["patch_size"],
        y_bins[0],
        y_bins[-1] + GEO_CFG["patch_size"],
    ]

    for npz_path in tqdm(npz_files, desc="Diagnostic Inference"):
        try:
            data = np.load(npz_path, allow_pickle=True)
            phys = torch.from_numpy(data["phys_8d"]).float().unsqueeze(0).to(DEVICE)
            geom = (
                torch.from_numpy(data[FEAT_CFG["3d"]["key_name"]])
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )
            tex = (
                torch.from_numpy(data[FEAT_CFG["2d"]["key_name"]])
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                logits, weights = model(phys, geom, tex)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                w_all = weights.squeeze(0).cpu().numpy()  # [N, 4]

            # 填充多维数据
            maps = {
                "prob": np.zeros((num_y, num_x)),
                "gt": np.zeros((num_y, num_x)),
                "dom": np.zeros((num_y, num_x)),
                "w3d": np.zeros((num_y, num_x)),
                "w2d": np.zeros((num_y, num_x)),
                "wsy": np.zeros((num_y, num_x)),
            }

            idx = 0
            for i in range(num_x):
                for j in range(num_y):
                    if idx < len(probs):
                        maps["prob"][j, i] = probs[idx]
                        maps["gt"][j, i] = data["meta"][idx, 0]
                        maps["dom"][j, i] = np.argmax(w_all[idx])
                        maps["w3d"][j, i] = w_all[idx, 1]  # 3D 权重
                        maps["w2d"][j, i] = w_all[idx, 2]  # 2D 权重
                        maps["wsy"][j, i] = w_all[idx, 3]  # Synergy 权重
                        idx += 1

            # 绘制 2x3 诊断矩阵
            fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
            plt.suptitle(f"Frame: {npz_path.stem} | Synergy Model v2.0", fontsize=16)

            # [0,0] 预测概率
            im00 = axes[0, 0].imshow(
                maps["prob"], cmap="jet", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 0].set_title("Detection Prob (MI-MoE)")
            fig.colorbar(im00, ax=axes[0, 0])

            # [0,1] 伪标签 (Ground Truth)
            im01 = axes[0, 1].imshow(
                maps["gt"], cmap="Reds", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 1].set_title("Pseudo-Label (Baseline)")

            # [0,2] 主导专家分布
            im02 = axes[0, 2].imshow(
                maps["dom"],
                cmap="terrain",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=3,
            )
            axes[0, 2].set_title("Dominant Expert")
            cbar02 = fig.colorbar(im02, ax=axes[0, 2], ticks=[0, 1, 2, 3])
            cbar02.ax.set_yticklabels(["Phys", "3D", "2D", "Syn"])

            # [1,0] 3D 专家权重 (证据1)
            im10 = axes[1, 0].imshow(
                maps["w3d"],
                cmap="Purples",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 0].set_title("3D Expert Weight")
            fig.colorbar(im10, ax=axes[1, 0])

            # [1,1] 2D 专家权重 (证据2)
            im11 = axes[1, 1].imshow(
                maps["w2d"],
                cmap="YlOrBr",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 1].set_title("2D Expert Weight")
            fig.colorbar(im11, ax=axes[1, 1])

            # [1,2] Synergy 专家权重 (核心证据)
            im12 = axes[1, 2].imshow(
                maps["wsy"],
                cmap="Greens",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 2].set_title("Synergy Interaction Weight")
            fig.colorbar(im12, ax=axes[1, 2])

            plt.savefig(vis_dir / f"diag_{npz_path.stem}.png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

    print(f"✅ 深度诊断报告已生成至: {vis_dir}")


if __name__ == "__main__":
    run_diagnostic_vis()
