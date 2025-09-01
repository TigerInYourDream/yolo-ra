#!/usr/bin/env python3
"""
实时监控训练进度
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os

def find_latest_run():
    """找到最新的训练目录"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    # 找到最新的目录
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "red-alert" in d.name]
    if not dirs:
        return None
    
    return max(dirs, key=os.path.getmtime)

def monitor_training():
    """监控训练过程"""
    print("🔍 寻找训练目录...")
    
    run_dir = find_latest_run()
    if not run_dir:
        print("❌ 没有找到训练目录")
        return
    
    csv_path = run_dir / "results.csv"
    print(f"📊 监控训练: {run_dir.name}")
    print(f"📈 TensorBoard: tensorboard --logdir {run_dir}")
    print("-" * 50)
    
    while True:
        try:
            if csv_path.exists():
                # 读取CSV
                df = pd.read_csv(csv_path)
                df.columns = [col.strip() for col in df.columns]
                
                # 显示最新的几行
                os.system('clear')  # 清屏
                print(f"📊 训练进度 - {run_dir.name}")
                print("=" * 60)
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    epoch = int(latest.get('epoch', 0))
                    
                    print(f"⏱️  Epoch: {epoch}")
                    print(f"📉 Box Loss: {latest.get('train/box_loss', 0):.4f}")
                    print(f"📉 Cls Loss: {latest.get('train/cls_loss', 0):.4f}")
                    print(f"📉 DFL Loss: {latest.get('train/dfl_loss', 0):.4f}")
                    
                    if 'metrics/mAP50(B)' in latest:
                        print(f"📈 mAP50: {latest.get('metrics/mAP50(B)', 0):.4f}")
                    if 'metrics/mAP50-95(B)' in latest:
                        print(f"📈 mAP50-95: {latest.get('metrics/mAP50-95(B)', 0):.4f}")
                    
                    print("\n最近5个Epoch:")
                    print("-" * 60)
                    
                    # 显示最近5行
                    recent = df.tail(5)[['epoch', 'train/box_loss', 'train/cls_loss']]
                    print(recent.to_string(index=False))
                
                print("\n按 Ctrl+C 退出监控")
                
            else:
                print("⏳ 等待训练开始...")
            
            time.sleep(2)  # 每2秒更新一次
            
        except KeyboardInterrupt:
            print("\n👋 退出监控")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(2)

if __name__ == "__main__":
    monitor_training()