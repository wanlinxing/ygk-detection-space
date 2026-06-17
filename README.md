---
title: 目标检测系统 - YOLOv8 + CBAM
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# 🔍 YOLOv8 目标检测系统

基于改进的 **YOLOv8** 模型实现 20 类目标检测，集成 **CBAM 注意力机制**、**Transformer 自注意力** 和 **Mamba 状态空间模型**。

## 🚀 在线体验

🌐 **Web 应用**: [https://wlx.ygk.wiki/](https://wlx.ygk.wiki/)

## 📖 项目概述

本项目使用 Pascal VOC 2012 数据集训练，可检测以下 20 类目标：

| 交通工具 | 动物 | 室内物品 | 其他 |
|---------|------|---------|------|
| ✈️ 飞机 | 🐦 鸟 | 🍶 瓶子 | 🧑 人 |
| 🚲 自行车 | 🐱 猫 | 🪑 椅子 | |
| 🚤 船 | 🐄 牛 | 🍽️ 餐桌 | |
| 🚌 公交车 | 🐶 狗 | 🪴 盆栽 | |
| 🚗 汽车 | 🐴 马 | 🛋️ 沙发 | |
| 🏍️ 摩托车 | 🐑 羊 | 📺 显示器 | |
| 🚆 火车 | | | |

## 📁 项目结构

```
深度学习/
├── app.py              # 🆕 Gradio Web 应用
├── config.py           # 配置文件
├── model.py            # YOLOv8 模型（含 CBAM/Transformer/Mamba）
├── dataset.py          # 数据集加载和预处理
├── loss.py             # 损失函数
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── download_data.py    # 数据集下载
├── utils/              # 🆕 工具模块
│   ├── __init__.py
│   ├── detector.py     # 检测器模块
│   └── visualize.py    # 可视化模块
├── runs/train/         # 训练结果和模型权重
├── requirements.txt    # 依赖包
└── README.md           # 项目说明
```

## 🔧 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860` 即可使用。

### 3. 训练模型

```bash
# 下载数据集
python download_data.py

# 开始训练
python train.py
```

## 🌐 部署

本项目已配置为可部署到 **Hugging Face Spaces**：

1. 将代码推送到 GitHub 仓库
2. 在 [Hugging Face Spaces](https://huggingface.co/spaces) 创建新 Space
3. 选择 Gradio SDK，关联 GitHub 仓库
4. Space 自动构建并部署

### 自定义域名配置

将域名 `wlx.ygk.wiki` 指向 HF Space：

1. 在 DNS 中添加 CNAME 记录：
   ```
   wlx.ygk.wiki  CNAME  wanlixue-目标检测.hf.space
   ```
2. 或使用 Cloudflare Tunnel / Nginx 反向代理

### 大型文件处理

模型权重文件 (~760MB) 使用 Git LFS 管理：

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pt" "*.pth" "*.png"
```

## 模型创新点

### 1. CBAM 注意力机制
在骨干网络 P3/P4/P5 层引入 CBAM（Convolutional Block Attention Module），提升特征表达能力。

### 2. Transformer 自注意力
在检测头引入 Multi-Head Self-Attention，捕捉全局上下文信息。

### 3. Mamba 状态空间模型
引入选择性状态空间模型（Selective State Space Model），实现高效的长序列建模。

## 实验结果

### 训练曲线
![训练曲线](runs/train/training_curves.png)

### 对比实验
![对比结果](comparison_results.png)

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选）

## 参考资料

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) (ECCV 2018)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

## 作者

- 课程：深度学习技术与应用
- 学期：2025-2026 学年春季学期
