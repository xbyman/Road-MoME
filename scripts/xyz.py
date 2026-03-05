"""
[Debug Tool] 空间坐标对齐校验 (全链路逻辑核查版 v1.5)
功能：
1. 空间对齐：在图像上叠加物理网格 ID。
2. 数据缺失核查：同步模拟伪标签逻辑，并在图上高亮“被脚本判定为异常”的格子。
3. 坐标顺序验证：对比 X-First 还是 Y-First 排序，定位热力图镜像或截断问题。

更新记录 (v1.5)：
- 修复重复匹配问题：通过 ID 唯一性过滤，确保每个 select_files 中的 ID 只处理一次。
"""

import cv2
import numpy as np
import yaml
import json
from pathlib import Path

# 加载配置
project_root = Path(__file__).resolve().parent.parent
with open(project_root / "config" / "config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

GEO = cfg["geometry"]
PATH = cfg["paths"]
INF_CFG = cfg.get("inference", {})


def verify_alignment():
    # 1. 准备物理参数
    roi_x = GEO["roi_x"]  # [-3.0, 3.0]
    roi_y = GEO["roi_y"]  # [-8.0, 0.0]
    p_size = GEO["patch_size"]
    step = p_size * (1 - GEO["overlap"])
    th_anomaly = GEO.get("th_anomaly", 0.035)

    # 模拟预处理时的网格生成逻辑 (关键：必须与 reconstruct_grid 顺序一致)
    # 当前逻辑：先遍历 X，再遍历 Y (X-First)
    x_bins = np.arange(roi_x[0], roi_x[1] - p_size + 0.1, step)
    y_bins = np.arange(roi_y[0], roi_y[1] - p_size + 0.1, step)

    output_dir = project_root / "data" / "debug_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 确定输入文件来源 (优先使用 select 中的样本)
    img_root = Path(PATH["raw_img_dir"])
    target_ids = INF_CFG.get("select_files", [])

    img_files = []
    if INF_CFG.get("mode") == "select" and target_ids:
        all_potential_imgs = list(img_root.rglob("*.jpg"))
        processed_ids = set()

        for tid in target_ids:
            if tid in processed_ids:
                continue

            # 查找匹配该 ID 的所有路径
            matches = [p for p in all_potential_imgs if tid in p.name]
            if matches:
                # [核心修复] 只选取第一个匹配项（通常是 left 文件夹），防止重复处理
                img_files.append(matches[0])
                processed_ids.add(tid)
            else:
                print(f"⚠️ 未能在目录中找到 ID 为 {tid} 的图片文件。")
    else:
        # 随机模式也进行去重保护
        all_potential_imgs = list(img_root.rglob("*.jpg"))
        unique_names = {}
        for p in all_potential_imgs:
            if p.name not in unique_names:
                unique_names[p.name] = p
        img_files = list(unique_names.values())[:3]

    print(f"🔍 正在执行全链路核查，样本数: {len(img_files)}")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 寻找对应的 NPZ 数据以对比真实的伪标签
        # 移除可能存在的文件名重复部分，精准匹配
        img_stem = img_path.stem
        npz_name = f"pkg_{img_stem}.npz"
        npz_path = Path(PATH["output_dir"]) / npz_name

        real_labels = None
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            real_labels = data["meta"][:, 0]  # 获取原始伪标签
            print(f"✅ 加载到关联数据包: {npz_name} (包含 {len(real_labels)} 个 Patch)")
        else:
            # 尝试模糊匹配，防止 pkg_ 前缀导致的匹配失败
            potential_npz = list(Path(PATH["output_dir"]).glob(f"*{img_stem}*.npz"))
            if potential_npz:
                npz_path = potential_npz[0]
                data = np.load(npz_path, allow_pickle=True)
                real_labels = data["meta"][:, 0]
                print(f"✅ 通过模糊匹配加载到数据包: {npz_path.name}")

        canvas = img.copy()
        idx = 0

        # 重复 reconstruct_grid 的双重循环逻辑
        for i, xi in enumerate(x_bins):
            for j, yi in enumerate(y_bins):
                # 计算图像上的像素坐标 (线性投影模拟)
                u1 = int((xi - roi_x[0]) / (roi_x[1] - roi_x[0]) * w)
                u2 = int((xi + p_size - roi_x[0]) / (roi_x[1] - roi_x[0]) * w)
                v1 = int((yi - roi_y[0]) / (roi_y[1] - roi_y[0]) * h)
                v2 = int((yi + p_size - roi_y[0]) / (roi_y[1] - roi_y[0]) * h)

                u1, u2 = max(0, u1), min(w, u2)
                v1, v2 = max(0, v1), min(h, v2)

                # 确定网格颜色
                color = (0, 255, 0)  # 默认绿色 (Normal)
                thickness = 1

                # 核心核查逻辑：如果 .npz 里说这个 ID 是异常，则画红框
                if real_labels is not None and idx < len(real_labels):
                    if real_labels[idx] > 0.5:
                        color = (0, 0, 255)  # 判定为病害，变红
                        thickness = 3
                elif real_labels is not None and idx >= len(real_labels):
                    color = (255, 0, 255)  # 紫色代表“由于点云稀疏，数据流在此处已断裂”
                    thickness = 2

                cv2.rectangle(canvas, (u1, v1), (u2, v2), color, thickness)
                # 在格子中心写 ID
                cv2.putText(
                    canvas,
                    f"{idx}",
                    (u1 + 10, v1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
                idx += 1

        # 增加统计栏
        info_txt = f"ID Total: {idx} | NPZ Patches: {len(real_labels) if real_labels is not None else 'N/A'}"
        cv2.putText(
            canvas,
            info_txt,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        save_path = output_dir / f"audit_{img_path.name}"
        cv2.imwrite(str(save_path), canvas)
        print(f"📊 审计图已生成: {save_path.name} | 红框=伪标签异常 | 紫框=数据缺失")


if __name__ == "__main__":
    verify_alignment()
