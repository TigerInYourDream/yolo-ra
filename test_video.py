#!/usr/bin/env python3
"""
测试视频检测
"""

from ultralytics import YOLO
import sys

# 加载模型
model = YOLO('runs/red-alert_20250901_001914/weights/best.pt')

# 视频路径
video_path = sys.argv[1] if len(sys.argv) > 1 else '~/Desktop/open-ra.mp4'

# 预测
print(f"🎯 正在检测: {video_path}")
results = model.predict(
    source=video_path,
    save=True,           # 保存结果
    conf=0.25,          # 置信度阈值
    save_txt=False,     # 不保存文本
    save_conf=True,     # 保存置信度
    show_labels=True,   # 显示标签
    show_conf=True,     # 显示置信度
    line_thickness=2,   # 线条粗细
)

print(f"✅ 检测完成！")
print(f"📁 结果保存在: runs/detect/")
print(f"🎬 打开查看: open runs/detect/predict*/")