"""
[Step 4] MoME 深度诊断可视化脚本 (v3.7 鲁棒性验证版)
核心功能：
1. 集成压力测试：在常规推理基础上，自动执行 2D 失效测试 (Zero-out Test)。
2. 权重漂移可视化：保留并整合 Expert Weight Drift 条形图，证明 log(q) 路由的有效性。
3. 2x4 全景布局：展示概率图、伪标签、正常专家分布、失效后专家分布以及各专家权重细节。
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

# 环境设置
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from models.mome_model import build_mome_model


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


def run_comprehensive_diagnostic():
    # 1. 加载模型 (v2.3 强制路由版)
    model_path = PATH_CFG["weights"]["mome_model"]
    if not os.path.exists(model_path):
        print(f"⚠️ 未找到权重文件: {model_path}，请先完成 v2.3 训练。")
        return

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
        npz_files = npz_files[: INF_CFG.get("batch_limit", 15)]

    vis_dir = Path(
        PATH_CFG.get("vis_output_dir", project_root / "data" / "inference_vis")
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 网格参数计算
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

    print(f"🚀 启动全景诊断 | 包含 Expert Weight Drift 验证功能")

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

            # --- 场景 1: 正常状态 (Normal Mode) ---
            q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()
            q_img_real = data.get("quality_2d", np.array([0.8]))[0]
            q_vec_normal = (
                torch.cat([q_geo, torch.full_like(q_geo, q_img_real)], dim=-1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                logits, w_normal, _ = model(phys, geom, tex, q_vec_normal)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                weights_normal = w_normal.squeeze(0).cpu().numpy()

            # --- 场景 2: 2D 失效压力测试 (Failure Mode) ---
            q_vec_fail = (
                torch.cat([q_geo, torch.full_like(q_geo, 0.05)], dim=-1)
                .unsqueeze(0)
                .to(DEVICE)
            )
            with torch.no_grad():
                _, w_fail, _ = model(phys, geom, tex * 0.0, q_vec_fail)  # 特征置零
                weights_fail = w_fail.squeeze(0).cpu().numpy()

            # --- 数据矩阵填充 ---
            maps = {
                "prob": np.zeros((num_y, num_x)),
                "gt": np.zeros((num_y, num_x)),
                "dom_norm": np.zeros((num_y, num_x)),
                "dom_fail": np.zeros((num_y, num_x)),
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
                        maps["dom_norm"][j, i] = np.argmax(weights_normal[idx])
                        maps["dom_fail"][j, i] = np.argmax(weights_fail[idx])
                        maps["w3d"][j, i] = weights_normal[idx, 1]
                        maps["w2d"][j, i] = weights_normal[idx, 2]
                        maps["wsy"][j, i] = weights_normal[idx, 3]
                        idx += 1

            # --- 绘图: 2x4 综合报告布局 ---
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
            plt.suptitle(
                f"MoME Robustness Diagnostic | Frame: {npz_path.stem}\nOriginal Img Quality (q2): {q_img_real:.3f}",
                fontsize=18,
                fontweight="bold",
            )

            # [0,0] 预测概率
            im00 = axes[0, 0].imshow(
                maps["prob"], cmap="jet", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 0].set_title("Detection Prob (Normal Mode)")
            fig.colorbar(im00, ax=axes[0, 0])

            # [0,1] 伪标签对比
            axes[0, 1].imshow(
                maps["gt"], cmap="Reds", extent=extent, origin="lower", vmin=0, vmax=1
            )
            axes[0, 1].set_title("Pseudo-Label (Baseline)")

            # [0,2] 主导专家 (正常状态)
            im02 = axes[0, 2].imshow(
                maps["dom_norm"],
                cmap="terrain",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=3,
            )
            axes[0, 2].set_title("Dominant Expert (Normal Mode)")
            cbar02 = fig.colorbar(im02, ax=axes[0, 2], ticks=[0, 1, 2, 3])
            cbar02.ax.set_yticklabels(["Phys", "3D", "2D", "Syn"])

            # [0,3] 主导专家 (2D 失效模拟)
            im03 = axes[0, 3].imshow(
                maps["dom_fail"],
                cmap="terrain",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=3,
            )
            axes[0, 3].set_title("Dominant Expert (Failure Mode: q2=0.05)")
            fig.colorbar(im03, ax=axes[0, 3], ticks=[0, 1, 2, 3])

            # [1,0] 保留核心功能: Expert Weight Drift Bar Chart
            labels = ["Phys", "3D-Geom", "2D-Tex", "Synergy"]
            avg_w_n = np.mean(weights_normal, axis=0)
            avg_w_f = np.mean(weights_fail, axis=0)
            x = np.arange(len(labels))
            axes[1, 0].bar(x - 0.2, avg_w_n, 0.4, label="Normal", color="skyblue")
            axes[1, 0].bar(x + 0.2, avg_w_f, 0.4, label="2D Fail", color="salmon")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(labels)
            axes[1, 0].set_title("Expert Weight Drift Analysis")
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1.0)
            axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

            # [1,1] 3D 权重分布细节
            im11 = axes[1, 1].imshow(
                maps["w3d"],
                cmap="Purples",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=0.1,
            )
            axes[1, 1].set_title("3D Weight Density (Normal)")
            fig.colorbar(im11, ax=axes[1, 1])

            # [1,2] 2D 权重分布细节
            im12 = axes[1, 2].imshow(
                maps["w2d"],
                cmap="YlOrBr",
                extent=extent,
                origin="lower",
                vmin=0,
                vmax=0.1,
            )
            axes[1, 2].set_title("2D Weight Density (Normal)")
            fig.colorbar(im12, ax=axes[1, 2])

            # [1,3] Synergy 权重分布细节
            im13 = axes[1, 3].imshow(
                maps["wsy"],
                cmap="Greens",
                extent=extent,
                origin="lower",
                vmin=0.8,
                vmax=1.0,
            )
            axes[1, 3].set_title("Synergy Core Weight (Normal)")
            fig.colorbar(im13, ax=axes[1, 3])

            plt.savefig(
                vis_dir / f"diag_robust_v37_{npz_path.stem}.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

        except Exception as e:
            print(f"❌ 帧 {npz_path.name} 诊断失败: {e}")

    print(f"✨ 全景诊断报告已生成至: {vis_dir}")


if __name__ == "__main__":
    run_comprehensive_diagnostic()
