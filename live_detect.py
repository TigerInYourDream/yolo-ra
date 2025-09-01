#!/usr/bin/env python3
"""
实时显示检测结果
"""

from ultralytics import YOLO
import cv2
import sys
from pathlib import Path

def detect_video(video_path, model_path='runs/red-alert_20250901_001914/weights/best.pt'):
    """实时检测并显示视频"""
    
    # 加载模型
    print(f"📦 加载模型...")
    model = YOLO(model_path)
    
    # 打开视频
    video_path = Path(video_path).expanduser()
    if not video_path.exists():
        print(f"❌ 文件不存在: {video_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎬 视频: {width}x{height} @ {fps}fps")
    print(f"🎮 按 'q' 退出, 空格暂停")
    
    # 创建窗口
    window_name = "红警单位检测 - 按Q退出"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 视频播放完毕")
                break
            
            frame_count += 1
            
            # YOLO检测
            results = model(frame, conf=0.25, verbose=False)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            
            # 显示统计信息
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            # 添加文字信息
            cv2.putText(annotated_frame, 
                       f"Frame: {frame_count} | Detections: {num_detections} | Press Q to quit", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # 显示帧
            cv2.imshow(window_name, annotated_frame)
        
        # 键盘控制
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'):  # 退出
            break
        elif key == ord(' '):  # 空格暂停
            paused = not paused
            if paused:
                print("⏸️  已暂停")
            else:
                print("▶️  继续播放")
        elif key == ord('s'):  # 保存当前帧
            cv2.imwrite(f'frame_{frame_count}.jpg', annotated_frame)
            print(f"💾 保存帧: frame_{frame_count}.jpg")
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 检测完成")

if __name__ == "__main__":
    # 默认视频路径
    video_path = sys.argv[1] if len(sys.argv) > 1 else "~/Desktop/openra.mp4"
    
    # 运行检测
    detect_video(video_path)