# 🎮 YOLO 红色警戒单位识别 (原生 MPS 加速版)

使用 YOLOv8 + Apple Silicon MPS 加速训练红色警戒游戏单位识别模型。

## ✨ 特性

- 🚀 **MPS 加速**：充分利用 M1/M2/M3 GPU 性能
- 📦 **uv 包管理**：超快的 Python 包管理器
- 🎯 **just 命令**：简洁的任务运行器（替代 Makefile）
- 🏷️ **数据标注**：集成 LabelMe 标注工具
- 🌐 **Web Demo**：Gradio 界面展示识别效果
- 📊 **10 类目标**：识别基地、兵营、坦克等红警单位

## 📋 系统要求

- macOS 12.0+ (Apple Silicon M1/M2/M3)
- Python 3.10+
- 至少 8GB 内存
- 10GB 可用磁盘空间

## 🚀 快速开始

### 1. 安装工具

```bash
# 安装 uv (Python 包管理器)
brew install uv

# 安装 just (命令运行器)
brew install just

# 查看可用命令
just
```

### 2. 初始化项目

```bash
# 一键初始化环境
just quickstart

# 或分步执行
just init        # 创建虚拟环境
just install     # 安装依赖
just dirs        # 创建目录结构
```

### 3. 检查环境

```bash
# 检查 MPS 支持
just check

# 查看系统信息
just info
```

## 📸 数据准备

### 1. 收集游戏截图

```bash
# 创建数据目录
just dirs

# 将游戏截图放入以下目录
datasets/red-alert/images/
```

建议收集：
- 不同地图场景
- 不同光照条件
- 各阵营单位
- 每类 100-200 张

### 2. 数据标注

```bash
# 启动 LabelMe 标注工具
just label

# 标注说明：
# 1. 打开图片
# 2. 创建矩形框 (Create Rectangle)
# 3. 输入类别名称
# 4. 保存为 JSON
```

类别列表：
- `allied_base` - 盟军基地
- `soviet_base` - 苏军基地
- `war_factory` - 战车工厂
- `barracks` - 兵营
- `refinery` - 矿场
- `power_plant` - 电厂
- `tank` - 坦克
- `infantry` - 步兵
- `aircraft` - 飞机
- `ore_truck` - 矿车

### 3. 转换标注格式

```bash
# LabelMe JSON → YOLO TXT
just convert-labels
```

## 🚂 模型训练

### 快速训练测试

```bash
# 10 轮快速测试
just train-quick
```

### 完整训练

```bash
# 默认 100 轮训练
just train

# 自定义参数
just train configs/red-alert.yaml yolov8n.pt 200
```

### 监控训练

```bash
# 启动 TensorBoard
just tensorboard

# 访问 http://localhost:6006
```

## 🎯 模型使用

### 推理测试

```bash
# 对测试集推理
just predict

# 指定模型和图片
just predict models/best.pt datasets/test/
```

### Web 演示

```bash
# 启动 Gradio 界面
just demo

# 访问 http://localhost:7860
```

### 实时检测

```bash
# 屏幕检测
just screen

# 摄像头检测（如果有）
just live
```

## 📊 性能参考

在 M2 Max 上的训练速度：

| 模型 | 数据量 | Batch Size | 训练时间 |
|------|--------|------------|----------|
| YOLOv8n | 1000张 | 16 | 2-3 小时 |
| YOLOv8s | 1000张 | 8 | 4-5 小时 |
| YOLOv8m | 1000张 | 4 | 6-8 小时 |

## 📁 项目结构

```
yolo-ra/
├── configs/          # 配置文件
│   └── red-alert.yaml
├── datasets/         # 数据集
│   └── red-alert/
│       ├── images/   # 图片
│       └── labels/   # 标注
├── scripts/          # 脚本
│   ├── train.py      # 训练脚本
│   ├── demo.py       # Web演示
│   └── ...
├── models/           # 保存的模型
├── runs/             # 训练结果
├── justfile          # 命令定义
├── pyproject.toml    # 项目配置
└── requirements.txt  # 依赖列表
```

## 🛠️ 常用命令

```bash
# 环境管理
just init          # 初始化环境
just install       # 安装依赖
just check         # 检查环境

# 数据处理
just label         # 标注工具
just stats         # 数据统计
just split-dataset # 分割数据集

# 训练相关
just train         # 开始训练
just train-quick   # 快速测试
just tensorboard   # 监控面板

# 推理演示
just predict       # 批量推理
just demo          # Web界面
just screen        # 屏幕检测

# 其他工具
just backup        # 备份模型
just clean         # 清理缓存
just format        # 格式化代码
```

## ⚠️ 注意事项

1. **MPS 限制**：
   - 不支持混合精度训练 (AMP)
   - 某些操作可能回退到 CPU
   - 首次运行需要编译优化

2. **内存管理**：
   - 大 batch size 可能导致内存不足
   - 建议监控 Activity Monitor

3. **数据质量**：
   - 标注准确性直接影响模型效果
   - 建议每类至少 100 张图片

## 🐛 故障排除

### MPS 不可用
```bash
# 检查 PyTorch 版本
uv pip show torch

# 重新安装
uv pip install --force-reinstall torch torchvision
```

### 训练速度慢
- 减小 batch size
- 使用 YOLOv8n (最小模型)
- 关闭数据增强

### 内存不足
- 减小图像尺寸 (640 → 416)
- 减小 batch size
- 使用 cache=False

## 📚 参考资源

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
- [LabelMe 使用指南](https://github.com/wkentaro/labelme)
- [Just 命令运行器](https://just.systems/)

## 📝 License

MIT License - 自由使用和修改

---

💡 **提示**：遇到问题？运行 `just info` 查看环境信息，或提交 Issue。