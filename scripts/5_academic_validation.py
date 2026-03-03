"""
[Step 5] MoME v2.2 学术验证脚本
功能：
1. 鲁棒性验证 (Zero-out Test): 模拟 2D 传感器完全失效，观察权重是否自动向 3D 补偿。
2. 灵敏度扫描 (q2-Sensitivity Scan): 模拟图像质量平滑变化，绘制门控网络的动态响应曲线。
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
    with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_config()
FEAT_CFG = cfg["features"]
PATH_CFG = cfg["paths"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_academic_tests(sample_id=None):
    # 1. 加载模型
    model = build_mome_model(cfg).to(DEVICE)
    model.load_state_dict(
        torch.load(PATH_CFG["weights"]["mome_model"], map_location=DEVICE)
    )
    model.eval()

    # 2. 获取测试样本
    npz_dir = Path(PATH_CFG["output_dir"])
    if sample_id:
        test_file = npz_dir / f"{sample_id}.npz"
    else:
        test_file = next(npz_dir.glob("*.npz"))

    data = np.load(test_file, allow_pickle=True)
    print(f"🧪 正在对样本执行学术诊断: {test_file.stem}")

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

    # --- 实验 1: Zero-out Test (2D 失效测试) ---
    print("▶ 正在执行 Experiment P0: Zero-out Test...")

    # 场景 A: 正常 (q2=0.8)
    q_vec_normal = (
        torch.cat([q_geo, torch.full_like(q_geo, 0.8)], dim=-1).unsqueeze(0).to(DEVICE)
    )
    _, w_normal = model(phys, geom, tex, q_vec_normal)

    # 场景 B: 2D 完全失效 (q2=0.0, tex=0)
    q_vec_fail = (
        torch.cat([q_geo, torch.full_like(q_geo, 0.0)], dim=-1).unsqueeze(0).to(DEVICE)
    )
    _, w_fail = model(phys, geom, tex * 0.0, q_vec_fail)

    # 统计病害区域的平均权重变化
    avg_w_normal = w_normal[0].mean(dim=0).cpu().detach().numpy()
    avg_w_fail = w_fail[0].mean(dim=0).cpu().detach().numpy()

    # 绘图：权重漂移轨迹图
    labels = ["Phys", "3D-Geom", "2D-Tex", "Synergy"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2, avg_w_normal, width, label="Normal (q2=0.8)", color="skyblue"
    )
    plt.bar(
        x + width / 2, avg_w_fail, width, label="2D Failed (q2=0.0)", color="salmon"
    )
    plt.ylabel("Expert Weight")
    plt.title("Expert Weight Drift under Modality Failure")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(project_root / "logs" / "zero_out_test.png")
    print(f"✅ 权重漂移图已生成: logs/zero_out_test.png")

    # --- 实验 2: q2-Sensitivity Scan (质量响应扫描) ---
    print("▶ 正在执行 Experiment P1: q2-Sensitivity Scan...")

    q2_steps = np.linspace(0.0, 1.0, 21)  # 0.0 到 1.0 扫描 20 个点
    weight_history = []

    for q2_val in q2_steps:
        q_vec_scan = (
            torch.cat([q_geo, torch.full_like(q_geo, q2_val)], dim=-1)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            _, weights = model(phys, geom, tex, q_vec_scan)
            avg_w = weights[0].mean(dim=0).cpu().numpy()
            weight_history.append(avg_w)

    weight_history = np.array(weight_history)  # [21, 4]

    plt.figure(figsize=(10, 6))
    colors = ["purple", "blue", "orange", "green"]
    for i in range(4):
        plt.plot(
            q2_steps,
            weight_history[:, i],
            label=f"w_{labels[i]}",
            color=colors[i],
            marker="o",
            markersize=4,
        )

    plt.xlabel("Input Image Quality (q2)")
    plt.ylabel("Gating Output Weight")
    plt.title("Gating Network Response to Image Quality Scan")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig(project_root / "logs" / "q2_sensitivity_scan.png")
    print(f"✅ 质量响应扫描曲线已生成: logs/q2_sensitivity_scan.png")


if __name__ == "__main__":
    # 你可以手动指定一个有病害的样本 ID，效果会更明显
    run_academic_tests()
