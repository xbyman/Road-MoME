"""
Road-MoME 实验记录管理器 (Experiment Manager)
功能：
1. 自动归档：每次训练生成唯一的 Experiment ID (时间戳)。
2. 配置备份：将当前的 config.yaml 复制一份到实验文件夹，防止后期修改配置导致无法回溯。
3. 结果摘要：自动生成 JSON 或 CSV 摘要，记录最终的 Loss、Acc 及模型路径。
4. 全局总表：在 logs 下维护一个实验索引表 (master_log.csv)，方便横向对比不同实验。
"""

import os
import json
import yaml
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    def __init__(self, project_root, experiment_name="MoME_Run"):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_id = f"{experiment_name}_{self.timestamp}"

        # 定义路径
        self.exp_dir = self.project_root / "logs" / "experiments" / self.exp_id
        self.master_log_path = self.project_root / "logs" / "experiments_master_log.csv"

        # 立即创建文件夹
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 实验存档已创建: {self.exp_dir}")

    def log_config(self, config_path):
        """备份当时的配置文件"""
        dest = self.exp_dir / "config_backup.yaml"
        shutil.copy2(config_path, dest)
        print(f"📜 配置文件已备份至实验目录。")

    def save_results(self, metrics, config_dict):
        """
        保存实验最终结果并更新全局总表
        metrics: dict, 例如 {'final_val_loss': 0.007, 'best_epoch': 45}
        """
        # 1. 保存单个实验的详细 JSON
        with open(self.exp_dir / "result_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        # 2. 构造一行总表记录
        # 提取 config 中最核心的几个参数，方便对比
        summary_row = {
            "exp_id": self.exp_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lr": config_dict["train"].get("lr"),
            "pos_weight": config_dict["train"].get("pos_weight"),
            "batch_size": config_dict["train"].get("batch_size"),
            "modal_mask": config_dict["train"].get("modal_mask_prob", 0.15),
            **metrics,  # 将结果指标并入
        }

        # 3. 更新全局 CSV 总表
        new_df = pd.DataFrame([summary_row])
        if self.master_log_path.exists():
            master_df = pd.read_csv(self.master_log_path)
            master_df = pd.concat([master_df, new_df], ignore_index=True)
        else:
            master_df = new_df

        master_df.to_csv(self.master_log_path, index=False)
        print(f"📊 实验指标已同步至全局总表: {self.master_log_path}")

    def get_exp_dir(self):
        return str(self.exp_dir)


if __name__ == "__main__":
    # 测试代码
    manager = ExperimentManager(".", "TestRun")
    manager.log_config("config/config.yaml")
    manager.save_results(
        {"val_loss": 0.0065, "val_acc": 0.98}, {"train": {"lr": 0.0001}}
    )
