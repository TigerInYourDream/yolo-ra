#!/usr/bin/env just --justfile

# 默认显示帮助
default:
    @just --list --unsorted

# 初始化环境
init:
    @echo "🚀 初始化环境..."
    uv venv
    source .venv/bin/activate && uv pip install ultralytics torch torchvision pillow numpy opencv-python matplotlib pyyaml tqdm
    @echo "✅ 环境初始化完成！"

# 训练模型
train epochs="200":
    @echo "🚂 开始训练模型 ({{epochs}} 轮)..."
    source .venv/bin/activate && python train_quick.py

# 测试模型
test:
    @echo "🎯 测试模型..."
    source .venv/bin/activate && python test_model.py

# Web演示
demo model="runs/*/weights/best.pt":
    @echo "🌐 启动 Web 演示..."
    source .venv/bin/activate && python scripts/demo.py --model {{model}}

# 预测
predict:
    @echo "📸 预测..."
    yolo predict model=runs/red-alert_20250901_001914/weights/best.pt source=/Users/xxxx/Desktop/openra.mp4 show=true conf=0.25

# 查看训练结果
results:
    @echo "📊 训练结果："
    @ls -la runs/
    @echo "\n查看结果图片："
    @echo "open runs/*/results.png"

# TensorBoard监控
monitor:
    @echo "📈 启动 TensorBoard..."
    source .venv/bin/activate && tensorboard --logdir runs

# 清理缓存
clean:
    rm -rf runs/* __pycache__ .pytest_cache
    @echo "✅ 清理完成"