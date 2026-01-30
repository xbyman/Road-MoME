"""
NPZ 特征包检查工具 (增强调试版)
功能：
1. 扫描 data/frame_packages 目录下的文件。
2. 详细展示指定 .npz 文件的内部 Keys、Shape 和数据类型。
3. 统计各特征的数值范围（自动处理并记录 NaN/Inf 异常）。
4. 针对物理特征 (phys_8d) 提供明细解析，定位数值爆炸来源。
"""

import os
import numpy as np
import yaml
from pathlib import Path

# 物理特征维度说明
PHYS_LABELS = [
    "最大高度差 (max_dz)",
    "异常点比例 (ratio)",
    "质心偏移-X (offset_x)",
    "质心偏移-Y (offset_y)",
    "线性度 (linear)",
    "高度方差 (z_var)",
    "高度标准差 (z_std)",
    "点云密度 (density)",
]


# ==================== 配置加载 ====================
def load_config():
    # 尝试从当前路径向上查找 config/config.yaml
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        print("❌ 找不到 config.yaml，请检查项目路径。")
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_file(file_path):
    print(f"\n" + "核心诊断报告".center(60, "="))
    print(f"🔍 文件名: {file_path.name}")
    print("=" * 60)

    try:
        # 加载数据包
        data = np.load(file_path, allow_pickle=True)
        keys = data.files
        print(f"📦 包含键名 (Keys): {keys}")

        for key in keys:
            val = data[key]
            print(f"\n--- 键: [{key}] ---")
            print(f"   - 维度 (Shape): {val.shape}")
            print(f"   - 类型 (Dtype): {val.dtype}")

            # 数值分析 (仅针对数值型数组)
            if np.issubdtype(val.dtype, np.number):
                # 检查特殊值
                has_nan = np.isnan(val).any()
                has_inf = np.isinf(val).any()

                # 安全计算统计量 (过滤掉非有限值)
                finite_mask = np.isfinite(val)
                if finite_mask.any():
                    v_min = np.min(val[finite_mask])
                    v_max = np.max(val[finite_mask])
                    v_mean = np.mean(val[finite_mask])
                    print(f"   - 有效值范围: [{v_min:.4f} ~ {v_max:.4f}]")
                    print(f"   - 有效值均值: {v_mean:.4f}")
                else:
                    print(f"   - ⚠️ 警告: 该数组中没有任何有效数值 (全部为 NaN/Inf)。")

                if has_nan or has_inf:
                    inf_count = np.isinf(val).sum()
                    nan_count = np.isnan(val).sum()
                    print(
                        f"   - 🚨 异常统计: 检测到 {inf_count} 个 Inf, {nan_count} 个 NaN"
                    )

                    # 如果是物理特征，定位具体的 Patch
                    if key == "phys_8d" and val.ndim == 2:
                        error_indices = np.where(~np.isfinite(val).all(axis=1))[0]
                        print(f"   - 📍 异常 Patch 索引: {error_indices.tolist()}")

            # 针对物理特征的详细明细 (显示第一行)
            if key == "phys_8d" and val.size > 0:
                print(f"   - 物理特征明细 (第一个 Patch):")
                sample = val[0] if val.ndim > 1 else val
                for i, label in enumerate(PHYS_LABELS):
                    if i < len(sample):
                        status = (
                            " [OK]" if np.isfinite(sample[i]) else " [!!! ERROR !!!]"
                        )
                        print(f"     {label}: {sample[i]:.4e}{status}")

        # 综合对齐验证
        print("\n" + "对齐验证".center(30, "-"))
        n_patches = data["phys_8d"].shape[0] if "phys_8d" in keys else 0
        has_3d = "deep_512d" in keys
        has_2d = "deep_2d_768d" in keys

        status_3d = "✅ 已注入" if has_3d else "❌ 缺失"
        status_2d = "✅ 已注入" if has_2d else "❌ 缺失"

        print(f"   - Patch 总数: {n_patches}")
        print(f"   - 3D 深度特征: {status_3d}")
        print(f"   - 2D 纹理特征: {status_2d}")

    except Exception as e:
        print(f"❌ 脚本读取过程出错: {e}")


def main():
    config = load_config()
    if not config:
        return

    pkg_dir = Path(config["paths"]["output_dir"])
    if not pkg_dir.exists():
        print(f"❌ 目录不存在: {pkg_dir}")
        return

    npz_files = sorted(list(pkg_dir.glob("*.npz")))
    if not npz_files:
        print(f"⚠️  目录为空。")
        return

    print(f"📁 发现 {len(npz_files)} 个特征包。")
    print("\n[1] 检查第一个文件\n[2] 检查最后一个文件\n[3] 检查随机一个文件\n[4] 退出")

    choice = input("\n请输入编号: ")
    if choice == "1":
        inspect_file(npz_files[0])
    elif choice == "2":
        inspect_file(npz_files[-1])
    elif choice == "3":
        inspect_file(npz_files[np.random.randint(0, len(npz_files))])
    else:
        print("程序结束。")


if __name__ == "__main__":
    main()
1