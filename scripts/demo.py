#!/usr/bin/env python3
"""
YOLO 红色警戒单位识别 - Gradio Web演示
"""

import gradio as gr
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import argparse
from pathlib import Path


class YOLODemo:
    def __init__(self, model_path):
        """初始化演示"""
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"🔧 使用设备: {self.device}")
        
        # 加载模型
        print(f"📦 加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 类别名称（红警单位）
        self.class_names = [
            '盟军基地', '苏军基地', '战车工厂', '兵营', '矿场',
            '电厂', '坦克', '步兵', '飞机', '矿车'
        ]
        
        # 类别颜色
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
        ]
    
    def detect(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """执行检测"""
        if image is None:
            return None, "请上传图片"
        
        # 运行推理
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device
        )
        
        # 绘制结果
        annotated = results[0].plot()
        
        # 统计检测结果
        detections = results[0].boxes
        stats = self._get_stats(detections)
        
        return Image.fromarray(annotated), stats
    
    def _get_stats(self, detections):
        """统计检测结果"""
        if detections is None or len(detections) == 0:
            return "未检测到任何单位"
        
        # 统计各类别数量
        class_counts = {}
        for box in detections:
            cls = int(box.cls)
            cls_name = self.class_names[cls] if cls < len(self.class_names) else f"类别{cls}"
            conf = float(box.conf)
            
            if cls_name not in class_counts:
                class_counts[cls_name] = []
            class_counts[cls_name].append(conf)
        
        # 生成统计信息
        stats_text = "📊 **检测结果统计：**\n\n"
        stats_text += f"检测到 **{len(detections)}** 个目标\n\n"
        
        for cls_name, confs in class_counts.items():
            avg_conf = sum(confs) / len(confs)
            stats_text += f"• **{cls_name}**: {len(confs)} 个 (平均置信度: {avg_conf:.2%})\n"
        
        return stats_text
    
    def batch_detect(self, files, conf_threshold=0.25, iou_threshold=0.45):
        """批量检测"""
        if not files:
            return None, "请上传图片"
        
        results_images = []
        all_stats = []
        
        for file in files:
            # 打开图片
            image = Image.open(file.name)
            
            # 运行检测
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device
            )
            
            # 绘制结果
            annotated = results[0].plot()
            results_images.append(Image.fromarray(annotated))
            
            # 统计
            detections = results[0].boxes
            stats = self._get_stats(detections)
            all_stats.append(f"**{Path(file.name).name}**\n{stats}")
        
        combined_stats = "\n\n---\n\n".join(all_stats)
        return results_images, combined_stats


def create_interface(model_path):
    """创建Gradio界面"""
    demo = YOLODemo(model_path)
    
    # 单张图片检测
    single_interface = gr.Interface(
        fn=demo.detect,
        inputs=[
            gr.Image(type="pil", label="上传游戏截图"),
            gr.Slider(0, 1, 0.25, label="置信度阈值"),
            gr.Slider(0, 1, 0.45, label="IOU阈值")
        ],
        outputs=[
            gr.Image(type="pil", label="检测结果"),
            gr.Markdown(label="统计信息")
        ],
        title="🎮 红色警戒单位检测 - 单张图片",
        description="上传红色警戒游戏截图，AI将自动识别其中的单位和建筑",
        examples=[
            ["examples/example1.jpg", 0.25, 0.45],
            ["examples/example2.jpg", 0.3, 0.5],
        ] if Path("examples").exists() else None,
        cache_examples=True if Path("examples").exists() else False,
    )
    
    # 批量检测
    batch_interface = gr.Interface(
        fn=demo.batch_detect,
        inputs=[
            gr.File(file_count="multiple", label="上传多张图片", file_types=["image"]),
            gr.Slider(0, 1, 0.25, label="置信度阈值"),
            gr.Slider(0, 1, 0.45, label="IOU阈值")
        ],
        outputs=[
            gr.Gallery(label="检测结果", columns=2),
            gr.Markdown(label="统计信息")
        ],
        title="🎮 红色警戒单位检测 - 批量处理",
        description="批量上传游戏截图进行检测",
    )
    
    # 实时检测（摄像头）
    live_interface = gr.Interface(
        fn=demo.detect,
        inputs=[
            gr.Image(source="webcam", type="pil", label="摄像头/屏幕捕获"),
            gr.Slider(0, 1, 0.25, label="置信度阈值"),
            gr.Slider(0, 1, 0.45, label="IOU阈值")
        ],
        outputs=[
            gr.Image(type="pil", label="检测结果"),
            gr.Markdown(label="统计信息")
        ],
        title="🎮 红色警戒单位检测 - 实时检测",
        description="使用摄像头或屏幕捕获进行实时检测",
        live=True,
    )
    
    # 组合界面
    demo_app = gr.TabbedInterface(
        [single_interface, batch_interface, live_interface],
        ["单张检测", "批量检测", "实时检测"],
        title="🎮 YOLO 红色警戒单位识别系统",
    )
    
    return demo_app


def main():
    parser = argparse.ArgumentParser(description='YOLO Web演示')
    parser.add_argument('--model', type=str, default='runs/train/exp/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--port', type=int, default=7860,
                        help='端口号')
    parser.add_argument('--share', action='store_true',
                        help='创建公共链接')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    # 创建并启动界面
    app = create_interface(args.model)
    
    print(f"🚀 启动 Web 界面...")
    print(f"📍 本地访问: http://localhost:{args.port}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        favicon_path=None,
    )


if __name__ == '__main__':
    main()