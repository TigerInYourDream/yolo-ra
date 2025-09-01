#!/usr/bin/env python3
"""
分割数据集为训练集、验证集和测试集
"""

import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """分割数据集"""
    
    # 确保比例和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例之和必须为1"
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # 获取所有图片
    image_dir = source_path / 'train' / 'images'
    label_dir = source_path / 'train' / 'labels'
    
    images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    print(f"找到 {len(images)} 张图片")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(images)
    
    # 计算分割点
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 分割数据
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    print(f"测试集: {len(test_images)} 张")
    
    # 复制文件
    for split_name, split_images in [('train', train_images), 
                                       ('valid', val_images), 
                                       ('test', test_images)]:
        
        img_dest = dest_path / split_name / 'images'
        lbl_dest = dest_path / split_name / 'labels'
        
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            # 复制图片
            shutil.copy2(img_path, img_dest / img_path.name)
            
            # 复制对应的标签
            label_name = img_path.stem + '.txt'
            label_path = label_dir / label_name
            
            if label_path.exists():
                shutil.copy2(label_path, lbl_dest / label_name)
            else:
                print(f"警告: 找不到标签文件 {label_name}")
    
    print("✅ 数据集分割完成！")
    
    # 创建新的data.yaml
    create_yaml(dest_path)

def create_yaml(dest_path):
    """创建YOLO配置文件"""
    
    yaml_content = f"""# YOLO 红色警戒数据集配置
path: {dest_path.absolute()}
train: train/images
val: valid/images
test: test/images

# 类别
nc: 7
names: ['bingying', 'dianchang', 'jidi', 'jungongchang', 'keji', 'kuangchang', 'leida']

# 中文名称对照
names_cn:
  bingying: '兵营'
  dianchang: '电厂'
  jidi: '基地'
  jungongchang: '军工厂'
  keji: '科技实验室'
  kuangchang: '矿场'
  leida: '雷达'
"""
    
    yaml_path = dest_path / 'data.yaml'
    yaml_path.write_text(yaml_content)
    print(f"✅ 配置文件已创建: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='分割数据集')
    parser.add_argument('--source', type=str, default='open-ra.v1i.yolov8',
                        help='源数据集目录')
    parser.add_argument('--dest', type=str, default='datasets/red-alert',
                        help='目标目录')
    parser.add_argument('--ratio', type=str, default='0.7:0.2:0.1',
                        help='训练:验证:测试 比例')
    
    args = parser.parse_args()
    
    # 解析比例
    ratios = [float(r) for r in args.ratio.split(':')]
    if len(ratios) == 2:
        train_ratio, val_ratio = ratios
        test_ratio = 0
    elif len(ratios) == 3:
        train_ratio, val_ratio, test_ratio = ratios
    else:
        raise ValueError("比例格式错误，应该是 train:val 或 train:val:test")
    
    split_dataset(args.source, args.dest, train_ratio, val_ratio, test_ratio)

if __name__ == '__main__':
    main()