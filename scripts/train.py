#!/usr/bin/env python3
"""
YOLO 红色警戒单位识别 - MPS加速训练脚本
"""

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import yaml
import time
from datetime import datetime


def check_mps():
    """检查MPS支持"""
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("⚠️ PyTorch 未启用 MPS 构建")
            return 'cpu'
        print("✅ MPS (Metal Performance Shaders) 可用")
        print(f"🖥️ 设备: {torch.backends.mps.is_available()}")
        return 'mps'
    else:
        print("⚠️ MPS 不可用，使用 CPU 训练")
        return 'cpu'


def train(args):
    """主训练函数"""
    
    # 检查设备
    device = check_mps() if args.device == 'auto' else args.device
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    print(f"📦 加载预训练模型: {args.model}")
    model = YOLO(args.model)
    
    # 训练参数
    train_params = {
        'data': args.config,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': device,
        'project': args.project,
        'name': args.name or f"red-alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': args.exist_ok,
        'pretrained': True,
        'optimizer': args.optimizer,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': args.resume,
        'amp': False if device == 'mps' else args.amp,  # MPS 不支持 AMP
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save': True,
        'save_period': -1,
        'cache': args.cache,
        'workers': args.workers,
        'patience': args.patience,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'plots': True,
    }
    
    # 开始训练
    print("🚂 开始训练...")
    print(f"📊 配置文件: {args.config}")
    print(f"🔢 训练轮数: {args.epochs}")
    print(f"📐 图像尺寸: {args.imgsz}")
    print(f"📦 批次大小: {args.batch}")
    
    start_time = time.time()
    
    # 训练
    results = model.train(**train_params)
    
    # 训练完成
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"✅ 训练完成！")
    print(f"⏱️ 用时: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"💾 模型保存位置: {args.project}/{train_params['name']}/weights/")
    print(f"📊 最佳模型: {args.project}/{train_params['name']}/weights/best.pt")
    print(f"📈 TensorBoard: tensorboard --logdir {args.project}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLO 红色警戒单位识别训练')
    
    # 基础参数
    parser.add_argument('--config', type=str, default='configs/red-alert.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='预训练模型 (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='训练图像尺寸')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备 (auto/mps/cpu)')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='优化器 (SGD/Adam/AdamW)')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率')
    
    # 训练控制
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--cache', type=str, default='ram',
                        help='数据缓存 (True/ram/disk/False)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--amp', action='store_true',
                        help='使用混合精度训练')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    
    # 保存参数
    parser.add_argument('--project', type=str, default='runs',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default=None,
                        help='实验名称')
    parser.add_argument('--exist-ok', action='store_true',
                        help='覆盖已存在的项目')
    
    args = parser.parse_args()
    
    # 执行训练
    train(args)


if __name__ == '__main__':
    main()