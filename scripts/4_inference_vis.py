"""
[Step 4] MoME 深度诊断可视化脚本 (v2.2 级联调制版)
核心更新：
1. 质量因子对齐：推理时自动从 NPZ 提取 quality_2d 与 quality_geo 构造 quality_vec。
2. 级联逻辑展示：展示经过 FiLM 校准与 Synergy 交互后的最终决策分布。
3. 颜色映射优化：维持 Phys, 3D, 2D, Synergy 四类专家的色块区分。
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

# 添加项目根目录到路径
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


def run_diagnostic_vis():
    # 1. 加载模型 (v2.2 级联版本)
    model_path = PATH_CFG["weights"]["mome_model"]
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到权重文件: {model_path}，请先运行 3_train_mome.py")
        return

    model = build_mome_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 准备数据
    npz_dir = Path(PATH_CFG["output_dir"])
    npz_files = sorted(list(npz_dir.glob("*.npz")))

    if not npz_files:
        print(f"❌ 未在 {npz_dir} 发现数据包")
        return

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

    # 3. 图像对齐准备 (用于后续扩展展示)
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

            # --- 构造模型输入 ---
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

            # [v2.2 核心] 提取质量因子向量
            q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()  # [N, 1]
            q_img_val = data.get("quality_2d", np.array([0.8]))[0]
            q_img = torch.full_like(q_geo, q_img_val)  # [N, 1]
            q_vec = (
                torch.cat([q_geo, q_img], dim=-1).unsqueeze(0).to(DEVICE)
            )  # [1, N, 2]

            with torch.no_grad():
                # 注意 v2.2 的输出不包含 gamma，如需展示 gamma 请在模型中返回
                logits, weights = model(phys, geom, tex, q_vec)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                w_all = weights.squeeze(0).cpu().numpy()  # [N, 4]

            # --- 填充多维热力图数据 ---
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

            # --- 绘制 2x3 诊断矩阵 ---
            fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
            plt.suptitle(
                f"Frame: {npz_path.stem} | Cascaded Synergy v2.2\nImg Quality (q2): {q_img_val:.3f}",
                fontsize=16,
            )

            # [0,0] 最终预测概率
            im00 = axes[0, 0].imshow(
                maps["prob"], cmap="jet", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 0].set_title("Detection Prob (Cascaded MoME)")
            fig.colorbar(im00, ax=axes[0, 0])

            # [0,1] 伪标签 (Baseline)
            im01 = axes[0, 1].imshow(
                maps["gt"], cmap="Reds", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 1].set_title("Pseudo-Label (Rule-based)")

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
            cbar02.ax.set_yticklabels(["Phys", "3D-Geom", "2D-Tex", "Synergy"])

            # [1,0] 3D 专家独立权重
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

            # [1,1] 2D 专家独立权重 (调制前)
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

            # [1,2] 协同专家权重 (级联后核心)
            im12 = axes[1, 2].imshow(
                maps["wsy"],
                cmap="Greens",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[1, 2].set_title("Synergy (FiLM+Inter) Weight")
            fig.colorbar(im12, ax=axes[1, 2])

            output_name = vis_dir / f"diag_v22_{npz_path.stem}.png"
            plt.savefig(output_name, bbox_inches="tight", dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error processing {npz_path.name}: {e}")
            continue

    print(f"✅ 级联诊断报告已生成至: {vis_dir}")


if __name__ == "__main__":
    run_diagnostic_vis()
