"""
Road-MoME 视觉真值标注工具 (v2.1.1 自动续标增强版)
功能：
1. 断点续标：自动读取已有 JSON，跳过已标注文件，重启后直接进入下一张。
2. 协作分工：支持通过 FOLDER_RANGE 指定时间戳文件夹范围。
3. 稳定按钮交互：UI 底部增加鼠标点击按钮，解决键盘不稳定的问题。
4. 视图锁定：精准匹配 {timestamp}/left/*.jpg 路径。
"""

import cv2
import numpy as np
import yaml
import json
import os
from pathlib import Path
from tqdm import tqdm

# ==================== 协作分工配置区 ====================
USER_NAME = "Annotator1"  # 标注员姓名 (用于区分保存文件)

# 指定分配给该标注员的时间戳文件夹索引范围 [start, end)
FOLDER_RANGE = [0, 20]
# ======================================================

# --- 1. 配置加载与环境准备 ---
project_root = Path(__file__).resolve().parent.parent
CONFIG_PATH = project_root / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RAW_IMG_DIR = Path(cfg["paths"]["raw_img_dir"])
NPZ_DIR = Path(cfg["paths"]["output_dir"])
SAVE_PATH = project_root / "data" / f"manual_visual_gt_{USER_NAME}.json"

ROI_X = cfg["geometry"]["roi_x"]
ROI_Y = cfg["geometry"]["roi_y"]
PATCH_SIZE = cfg["geometry"]["patch_size"]
OVERLAP = cfg["geometry"]["overlap"]


class ProfessionalAnnotator:
    def __init__(self):
        # [核心功能] 1. 加载已有标注实现“自动续标”
        self.all_manual_labels = {}
        if SAVE_PATH.exists():
            with open(SAVE_PATH, "r", encoding="utf-8") as f:
                self.all_manual_labels = json.load(f)
            print(
                f"📂 [{USER_NAME}] 检测到已有存档，已加载 {len(self.all_manual_labels)} 帧记录。"
            )

        # 2. 建立特征包索引 (确保只标注有 3D 数据支持的帧)
        print("🔍 正在扫描特征包索引...")
        self.available_npz_stems = {
            p.stem.replace("pkg_", "") for p in NPZ_DIR.glob("*.npz")
        }

        # 3. 任务分配与过滤
        all_timestamp_dirs = sorted(
            [d for d in RAW_IMG_DIR.iterdir() if d.is_dir() and (d / "left").exists()]
        )

        total_folders = len(all_timestamp_dirs)
        start, end = FOLDER_RANGE
        end = min(end, total_folders)
        my_dirs = all_timestamp_dirs[start:end]

        # [核心功能] 4. 构建 Todo 列表：跳过已存在于存档中的文件名
        self.todo_list = []
        for d in my_dirs:
            left_dir = d / "left"
            imgs = sorted(list(left_dir.glob("*.jpg")))
            for p in imgs:
                # 检查：1. 是否有对应的 NPZ 包 2. 是否还没有标过
                if p.stem in self.available_npz_stems:
                    if p.name not in self.all_manual_labels:
                        self.todo_list.append(p)

        print(f"📊 任务分配报告:")
        print(f"   - 负责目录范围: [{start}:{end}] (共 {len(my_dirs)} 个目录)")
        print(f"   - 已标注数量: {len(self.all_manual_labels)} 帧")
        print(f"   - 剩余待标注: {len(self.todo_list)} 帧 (重启后已自动跳转到断点)")

        # 5. 几何参数计算
        step = PATCH_SIZE * (1 - OVERLAP)
        self.x_bins = np.arange(ROI_X[0], ROI_X[1] - PATCH_SIZE + 0.1, step)
        self.y_bins = np.arange(ROI_Y[0], ROI_Y[1] - PATCH_SIZE + 0.1, step)
        self.num_patches = len(self.x_bins) * len(self.y_bins)

        # 状态控制
        self.current_labels = [0] * self.num_patches
        self.is_drawing = False
        self.draw_mode = 1
        self.should_save = False
        self.should_quit = False

        # UI 按钮布局
        self.footer_h = 80
        self.buttons = {
            "save": [20, 10, 220, 60, "SAVE & NEXT"],
            "clear": [260, 10, 120, 60, "CLEAR"],
            "quit": [400, 10, 120, 60, "QUIT"],
        }

    def get_patch_idx(self, x, y, w, h):
        """物理坐标映射至网格索引"""
        if y > h:
            return None
        idx = 0
        for xi in self.x_bins:
            for yi in self.y_bins:
                u1 = int((xi - ROI_X[0]) / (ROI_X[1] - ROI_X[0]) * w)
                u2 = int((xi + PATCH_SIZE - ROI_X[0]) / (ROI_X[1] - ROI_X[0]) * w)
                v1 = int((yi - ROI_Y[0]) / (ROI_Y[1] - ROI_Y[0]) * h)
                v2 = int((yi + PATCH_SIZE - ROI_Y[0]) / (ROI_Y[1] - ROI_Y[0]) * h)
                if u1 <= x <= u2 and v1 <= y <= v2:
                    return idx
                idx += 1
        return None

    def on_mouse(self, event, x, y, flags, param):
        img_h = param["h"]
        img_w = param["w"]

        if event == cv2.EVENT_LBUTTONDOWN:
            if y > img_h:  # 点击按钮区
                local_y = y - img_h
                for btn_id, (bx, by, bw, bh, _) in self.buttons.items():
                    if bx <= x <= bx + bw and by <= local_y <= by + bh:
                        if btn_id == "save":
                            self.should_save = True
                        if btn_id == "clear":
                            self.current_labels = [0] * self.num_patches
                        if btn_id == "quit":
                            self.should_quit = True
                return

            idx = self.get_patch_idx(x, y, img_w, img_h)
            if idx is not None:
                self.is_drawing = True
                self.draw_mode = 1 - self.current_labels[idx]
                self.current_labels[idx] = self.draw_mode

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            if y <= img_h:
                idx = self.get_patch_idx(x, y, img_w, img_h)
                if idx is not None:
                    self.current_labels[idx] = self.draw_mode

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False

    def render(self, img):
        h, w = img.shape[:2]
        canvas = np.zeros((h + self.footer_h, w, 3), dtype=np.uint8)
        canvas[:h, :w] = img.copy()

        idx = 0
        grid_layer = canvas[:h, :w].copy()
        for xi in self.x_bins:
            for yi in self.y_bins:
                u1 = int((xi - ROI_X[0]) / (ROI_X[1] - ROI_X[0]) * w)
                u2 = int((xi + PATCH_SIZE - ROI_X[0]) / (ROI_X[1] - ROI_X[0]) * w)
                v1 = int((yi - ROI_Y[0]) / (ROI_Y[1] - ROI_Y[0]) * h)
                v2 = int((yi + PATCH_SIZE - ROI_Y[0]) / (ROI_Y[1] - ROI_Y[0]) * h)

                color = (0, 0, 255) if self.current_labels[idx] == 1 else (0, 255, 0)
                alpha = 0.35 if self.current_labels[idx] == 1 else 0.1
                cv2.rectangle(grid_layer, (u1, v1), (u2, v2), color, -1)
                cv2.rectangle(canvas[:h, :w], (u1, v1), (u2, v2), color, 1)
                idx += 1
        cv2.addWeighted(grid_layer, 0.4, canvas[:h, :w], 0.6, 0, canvas[:h, :w])

        footer = canvas[h:, :]
        footer[:] = (45, 45, 45)
        for btn_id, (bx, by, bw, bh, label) in self.buttons.items():
            color = (60, 160, 60) if btn_id == "save" else (100, 100, 100)
            if btn_id == "quit":
                color = (60, 60, 160)
            cv2.rectangle(footer, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.putText(
                footer,
                label,
                (bx + 15, by + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # 统计进度显示
        done_count = len(self.all_manual_labels)
        total_todo = done_count + len(self.todo_list)
        cv2.putText(
            canvas,
            f"Progress: {done_count}/{total_todo} | User: {USER_NAME}",
            (w - 380, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return canvas

    def run(self):
        if not self.todo_list:
            print(f"🎉 任务已全部完成！")
            return

        win_name = f"Road-MoME Annotator - [{USER_NAME}]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 850)

        for img_path in self.todo_list:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            self.current_labels = [0] * self.num_patches
            self.should_save = False

            cv2.setMouseCallback(win_name, self.on_mouse, {"w": w, "h": h})

            while True:
                display = self.render(img)
                cv2.imshow(win_name, display)

                cv2.waitKey(10)

                if self.should_save:
                    # 将当前帧存入字典并同步到磁盘
                    self.all_manual_labels[img_path.name] = self.current_labels
                    with open(SAVE_PATH, "w", encoding="utf-8") as f:
                        json.dump(self.all_manual_labels, f)
                    print(f"✔️ Saved: {img_path.name}")
                    break

                if self.should_quit:
                    print("🚪 安全退出，进度已保存。")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("✨ 标注流程已圆满结束。")


if __name__ == "__main__":
    ProfessionalAnnotator().run()
