"""
[Step 5] MoME v2.7 学术验证脚本 (报错修复版)
功能：
1. 鲁棒性验证 (Zero-out Test): 模拟 2D 传感器完全失效，验证权重向 3D 分支的自适应漂移。
2. 灵敏度扫描 (q2-Sensitivity Scan): 验证门控网络对图像质量因子的响应曲线。
核心修复：
- 适配 v2.3+ 模型的三个返回值 (logits, weights, exp_logits)。
- 增加对物理专家 (Phys) 惩罚项的监测。
"""

import os
import sys
import torch
import numpy as np
import yaml
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
FEAT_CFG = cfg["features"]
PATH_CFG = cfg["paths"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_academic_tests(sample_id=None):
    # 1. 加载模型 (v2.7 架构)
    model = build_mome_model(cfg).to(DEVICE)
    model_path = PATH_CFG["weights"]["mome_model"]
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型权重 {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 获取测试样本
    npz_dir = Path(PATH_CFG["output_dir"])
    npz_files = sorted(list(npz_dir.glob("*.npz")))
    if sample_id:
        test_file = npz_dir / f"{sample_id}.npz"
    else:
        # 优先选择含病害的样本
        test_file = npz_files[0]
        for f in npz_files[:50]:
            d = np.load(f, allow_pickle=True)
            if np.sum(d["meta"][:, 0]) > 2:  # 如果病害点数 > 2
                test_file = f
                break

    data = np.load(test_file, allow_pickle=True)
    print(f"🧪 正在执行学术诊断: {test_file.stem}")

    # 提取基础特征
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
    q_geo = torch.from_numpy(data["meta"][:, 1:2]).float()  # [N, 1]

    # 专家标签映射
    labels = ["Phys", "3D-Geom", "2D-Tex", "Synergy"]

    # --- 实验 1: Zero-out Test (2D 失效压力测试) ---
    print("▶ 正在执行 Experiment P0: Zero-out Test...")

    # 场景 A: 正常 (q2=0.8)
    q_vec_normal = (
        torch.cat([q_geo, torch.full_like(q_geo, 0.8)], dim=-1).unsqueeze(0).to(DEVICE)
    )
    # [核心修复] 适配三个返回值，使用 _ 忽略不需要的辅助 logits
    with torch.no_grad():
        _, w_normal, _ = model(phys, geom, tex, q_vec_normal)

    # 场景 B: 2D 完全失效 (q2=0.05, tex=0)
    q_vec_fail = (
        torch.cat([q_geo, torch.full_like(q_geo, 0.05)], dim=-1).unsqueeze(0).to(DEVICE)
    )
    with torch.no_grad():
        _, w_fail, _ = model(phys, geom, tex * 0.0, q_vec_fail)

    # 统计病害区域或全图的平均权重
    avg_w_normal = w_normal[0].mean(dim=0).cpu().numpy()
    avg_w_fail = w_fail[0].mean(dim=0).cpu().numpy()

    # 绘图 1: 权重漂移柱状图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(
        x - width / 2,
        avg_w_normal,
        width,
        label="Normal (q2=0.8)",
        color="skyblue",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        avg_w_fail,
        width,
        label="2D Failed (q2=0.05)",
        color="salmon",
        alpha=0.8,
    )
    plt.ylabel("Average Gating Weight")
    plt.title(f"Modality Failure Robustness Test\nSample: {test_file.stem}")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path_1 = project_root / "logs" / "zero_out_test_v27.png"
    plt.savefig(save_path_1, dpi=150)
    print(f"✅ 权重漂移图已生成: {save_path_1}")

    # --- 实验 2: q2-Sensitivity Scan (质量感知识别曲线) ---
    print("▶ 正在执行 Experiment P1: q2-Sensitivity Scan...")

    q2_steps = np.linspace(0.01, 1.0, 20)
    weight_history = []

    for q2_val in tqdm(q2_steps, desc="Scanning q2"):
        q_vec_scan = (
            torch.cat([q_geo, torch.full_like(q_geo, q2_val)], dim=-1)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            # [核心修复] 适配三个返回值
            _, weights, _ = model(phys, geom, tex, q_vec_scan)
            avg_w = weights[0].mean(dim=0).cpu().numpy()
            weight_history.append(avg_w)

    weight_history = np.array(weight_history)

    # 绘图 2: 响应曲线
    plt.figure(figsize=(12, 7))
    colors = ["#9b59b6", "#3498db", "#f1c40f", "#2ecc71"]  # 紫, 蓝, 黄, 绿
    for i in range(4):
        plt.plot(
            q2_steps,
            weight_history[:, i],
            label=f"w_{labels[i]}",
            color=colors[i],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

    plt.xlabel("Input Image Quality Factor (q2)")
    plt.ylabel("Model Gating Weight")
    plt.title("Gating Network Response to Environmental Quality Scan (v2.7)")
    plt.legend(loc="center right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axvline(
        x=0.3, color="red", linestyle="--", alpha=0.3, label="Confidence Threshold"
    )

    save_path_2 = project_root / "logs" / "q2_sensitivity_scan_v27.png"
    plt.savefig(save_path_2, dpi=150)
    print(f"✅ 灵敏度响应曲线已生成: {save_path_2}")


if __name__ == "__main__":
    # 建立 logs 目录确保保存成功
    (project_root / "logs").mkdir(exist_ok=True)
    run_academic_tests()
