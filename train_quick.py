#!/usr/bin/env python3
"""
快速训练脚本 - 使用 Roboflow 数据集
"""

from ultralytics import YOLO
import torch

# 检查设备
if torch.backends.mps.is_available():
    device = 'mps'
    print("✅ 使用 Apple Silicon GPU (MPS) 加速")
else:
    device = 'cpu'
    print("⚠️ 使用 CPU 训练")

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用最小的模型快速测试

# 训练参数（针对小数据集优化）
results = model.train(
    data='datasets/red-alert/data.yaml',  # 数据集配置
    epochs=200,  # 增加到200轮，获得更好效果
    imgsz=640,  # 图像大小
    batch=8,  # 批次大小（数据少，用小batch）
    device=device,  # 使用MPS加速
    patience=20,  # 早停耐心值
    save=True,  # 保存模型
    plots=True,  # 生成图表
    project='runs/train',  # 保存路径
    name='red-alert-v1',  # 实验名称
    exist_ok=True,  # 覆盖已存在的
    
    # 数据增强（适度）
    hsv_h=0.015,  # 色调变化
    hsv_s=0.7,    # 饱和度变化
    hsv_v=0.4,    # 明度变化
    degrees=0.0,  # 旋转角度
    translate=0.1,  # 平移
    scale=0.5,  # 缩放
    fliplr=0.5,  # 水平翻转
    mosaic=1.0,  # Mosaic增强
    
    # 优化器设置
    optimizer='SGD',
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    # 损失权重
    box=7.5,
    cls=0.5,
)

print("\n✅ 训练完成！")
print(f"📊 模型保存位置: runs/train/red-alert-v1/weights/")
print(f"   - 最佳模型: runs/train/red-alert-v1/weights/best.pt")
print(f"   - 最后模型: runs/train/red-alert-v1/weights/last.pt")
print("\n下一步：")
print("1. 查看训练结果: tensorboard --logdir runs/train")
print("2. 测试模型: python test_model.py")
print("3. Web演示: just demo")