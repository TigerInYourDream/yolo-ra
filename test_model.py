#!/usr/bin/env python3
"""
测试训练好的模型
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# 检查模型文件
model_path = Path("runs/train/red-alert-v1/weights/best.pt")
if not model_path.exists():
    print("❌ 模型文件不存在，请先训练模型")
    print("运行: python3 train_quick.py")
    exit(1)

# 加载模型
print(f"📦 加载模型: {model_path}")
model = YOLO(model_path)

# 设备选择
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🔧 使用设备: {device}")

# 在测试集上评估
print("\n📊 在测试集上评估...")
metrics = model.val(
    data='datasets/red-alert/data.yaml',
    split='test',
    device=device
)

# 打印评估结果
print("\n📈 评估结果:")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# 对测试图片进行推理
test_images = Path("datasets/red-alert/test/images")
if test_images.exists():
    print("\n🎯 对测试图片进行推理...")
    results = model.predict(
        source=test_images,
        save=True,
        save_txt=True,
        conf=0.25,
        device=device,
        project='runs/predict',
        name='test_results',
        exist_ok=True
    )
    print(f"✅ 预测结果保存在: runs/predict/test_results/")
    
# 类别性能
print("\n📊 各类别性能:")
names = model.names
for i, name in names.items():
    if i < len(metrics.box.ap50):
        ap = metrics.box.ap50[i]
        print(f"  {name}: AP50={ap:.3f}")

print("\n💡 提示:")
print("- 查看预测结果: open runs/predict/test_results/")
print("- 启动Web演示: python3 scripts/demo.py --model runs/train/red-alert-v1/weights/best.pt")