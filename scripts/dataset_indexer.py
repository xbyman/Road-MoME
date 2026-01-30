"""
数据集索引工具 (Dataset Indexer) - 多模态对齐版
功能：
1. 递归扫描原始目录下的所有时间戳子文件夹。
2. 自动匹配 PCD (3D) 与 JPG (2D) 文件对。
   匹配规则：.../{timestamp}/pcd/name.pcd <-> .../{timestamp}/left/name.jpg
3. 确保样本的完整性，过滤掉缺失模态的样本。
4. [新增] 将对齐后的路径列表保存为 data/dataset_index.yaml 供后续训练直接调用。
"""

import os
from pathlib import Path
import yaml


def generate_index():
    # 1. 加载配置
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ 错误: 找不到 config/config.yaml")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 获取原始数据根目录
    raw_root = Path(config["paths"]["raw_pcd_dir"])

    # 定义索引保存路径
    index_save_path = (
        Path(config["paths"].get("output_dir", "data/frame_packages")).parent
        / "dataset_index.yaml"
    )
    index_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔍 正在扫描原始目录以匹配多模态数据对: {raw_root}")

    # 2. 搜索逻辑：先找到所有 PCD
    # 路径示例: .../{timestamp}/pcd/20230317074844.400.pcd
    all_pcd_paths = list(raw_root.glob("**/pcd/*.pcd"))

    valid_pairs = []
    missing_count = 0

    # 3. 寻找匹配的 2D 图像
    # 对应路径: .../{timestamp}/left/20230317074844.400.jpg
    for pcd_path in all_pcd_paths:
        # pcd_path.parent 是 'pcd' 文件夹
        # pcd_path.parent.parent 是 '{timestamp}' 文件夹
        timestamp_dir = pcd_path.parent.parent
        img_name = f"{pcd_path.stem}.jpg"

        # 构造预期的图像路径：在同级的 left 文件夹下
        img_path = timestamp_dir / "left" / img_name

        if img_path.exists():
            # 记录绝对路径，方便跨目录脚本调用
            valid_pairs.append(
                {
                    "pcd": str(pcd_path.resolve()),
                    "img": str(img_path.resolve()),
                    "timestamp": timestamp_dir.name,
                    "id": pcd_path.stem,
                }
            )
        else:
            missing_count += 1

    # 4. 打印汇总报告并保存
    print("\n" + "=" * 50)
    print(f"📊 数据对齐报告:")
    print(f"   - 总计发现 PCD 数量: {len(all_pcd_paths)}")
    print(f"   - 成功配对 (3D+2D): {len(valid_pairs)}")
    print(f"   - 缺失图像的样本: {missing_count}")
    print("=" * 50)

    if valid_pairs:
        # 保存索引文件，后续脚本只需加载此 YAML
        with open(index_save_path, "w", encoding="utf-8") as f:
            yaml.dump(valid_pairs, f, allow_unicode=True)

        print(f"✅ 索引已保存至: {index_save_path}")
        print(f"\n💡 样例路径对比:")
        print(f"   [3D]: {valid_pairs[0]['pcd']}")
        print(f"   [2D]: {valid_pairs[0]['img']}")

    return valid_pairs


if __name__ == "__main__":
    pairs = generate_index()
