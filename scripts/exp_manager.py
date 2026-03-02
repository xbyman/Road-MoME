"""
Road-MoME 实验记录管理器 (Experiment Manager) - 多维指标增强版
功能：
1. 自动归档：每次训练生成唯一的 Experiment ID (时间戳)。
2. 配置备份：备份 config.yaml，防止参数修改导致无法回溯。
3. 结果摘要：支持记录 F1、Precision、Recall 及专家贡献率等多维指标。
4. 全局总表：在 logs 下维护实验索引表，方便横向对比不同实验的“病害召回能力”。
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
        print(f"📁 实验存档文件夹已建立: {self.exp_dir}")

    def log_config(self, config_path):
        """备份当前的配置文件"""
        if not os.path.exists(config_path):
            print(f"⚠️ 警告: 找不到配置文件 {config_path}，无法备份。")
            return
        dest = self.exp_dir / "config_backup.yaml"
        shutil.copy2(config_path, dest)
        print(f"📜 配置文件已备份至实验目录。")

    def save_results(self, metrics, config_dict):
        """
        保存最终结果并更新全局索引表
        metrics: dict，建议包含 {'best_val_loss', 'f1_score', 'recall', 'precision', 'avg_w_phys', ...}
        """
        # 1. 保存详细 JSON
        with open(self.exp_dir / "result_summary.json", "w", encoding="utf-8") as f:
            # 转换 metrics 中可能的 numpy 类型为原生 Python 类型以支持 JSON 序列化
            serializable_metrics = {
                k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()
            }
            json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)

        # 2. 构造总表记录 (提取核心参数与关键评价指标)
        summary_row = {
            "exp_id": self.exp_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lr": config_dict["train"].get("lr"),
            "pos_weight": config_dict["train"].get("pos_weight"),
            "modal_mask": config_dict["train"].get("modal_mask_prob"),
            # 核心评价指标
            "val_loss": metrics.get("best_val_loss"),
            "f1": round(metrics.get("f1_score", 0.0), 4),
            "recall": round(metrics.get("recall", 0.0), 4),
            "precision": round(metrics.get("precision", 0.0), 4),
            # 专家分布统计
            "w_phys": round(metrics.get("avg_w_phys", 0.0), 3),
            "w_geom": round(metrics.get("avg_w_geom", 0.0), 3),
            "w_tex": round(metrics.get("avg_w_tex", 0.0), 3),
        }

        # 3. 更新全局 CSV
        new_df = pd.DataFrame([summary_row])
        if self.master_log_path.exists():
            try:
                master_df = pd.read_csv(self.master_log_path)
                master_df = pd.concat([master_df, new_df], ignore_index=True)
            except Exception:
                master_df = new_df
        else:
            master_df = new_df

        master_df.to_csv(self.master_log_path, index=False)
        print(f"📊 实验多维指标已同步至全局总表: {self.master_log_path}")

    def get_exp_dir(self):
        return str(self.exp_dir)
