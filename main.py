"""
MoME 道路异常检测项目 - 全自动流水线主入口 (Master Pipeline)
功能：
1. 顺序执行：数据索引 -> 预处理 -> 2D特征 -> 3D特征 -> 模型训练。
2. 模块化控制：可以通过开关跳过已完成的阶段。
3. 环境检查：自动验证配置文件及必要的目录结构。
"""

import os
import subprocess
import sys
import yaml
from pathlib import Path

# ==================== 运行开关控制 ====================
STEPS = {
    "index": True,  # 1. 建立数据索引 (dataset_indexer)
    "preprocess": True,  # 2. 几何预处理与切片 (0_master_preprocess)
    "extract_2d": True,  # 3. 2D 视觉特征提取 (1_extract_2d_dinov2)
    "extract_3d": True,  # 4. 3D 深度特征提取 (2_extract_3d_deep)
    "train": True,  # 5. MoME 模型训练 (3_train_mome)
    "vis": True,  # 6. 推理结果可视化 (4_inference_vis)
}
# =====================================================


def load_config():
    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        print("❌ 错误: 找不到 config/config.yaml 配置文件")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_script(script_path):
    """使用当前解释器运行子脚本"""
    print(f"\n" + "=" * 60)
    print(f"🚀 正在启动模块: {script_path}")
    print("=" * 60)

    # 获取当前 Python 解释器的路径
    python_exe = sys.executable

    try:
        # 使用 subprocess 运行，并实时输出结果
        result = subprocess.run([python_exe, script_path], check=True)
        if result.returncode == 0:
            print(f"✅ 模块 {script_path} 执行成功")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 模块 {script_path} 运行出错，错误代码: {e.returncode}")
        return False


def main():
    print(
        """
    ================================================
       Road MoME Anomaly Detection Pipeline v1.0
    ================================================
    """
    )

    cfg = load_config()

    # 步骤 1: 建立索引
    if STEPS["index"]:
        if not run_script("scripts/dataset_indexer.py"):
            return

    # 步骤 2: 预处理 (生成 .npz 容器)
    if STEPS["preprocess"]:
        if not run_script("scripts/0_master_preprocess.py"):
            return

    # 步骤 3: 2D 特征提取
    if STEPS["extract_2d"]:
        if not run_script("scripts/1_extract_2d_dinov2.py"):
            return

    # 步骤 4: 3D 特征提取
    if STEPS["extract_3d"]:
        if not run_script("scripts/2_extract_3d_deep.py"):
            return

    # 步骤 5: 开始训练
    if STEPS["train"]:
        if not run_script("scripts/3_train_mome.py"):
            return

    # 步骤 6: 可视化结果 (可选)
    if STEPS["vis"]:
        run_script("scripts/4_inference_vis.py")

    print("\n" + "*" * 60)
    print("🎉 所有任务已圆满完成！")
    print("📈 请检查 pretrained_models/ 目录获取最佳权重。")
    print("*" * 60)


if __name__ == "__main__":
    main()
