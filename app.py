"""
目标检测 Web 应用 - Gradio 界面
YOLOv8 + CBAM + Transformer/Mamba 改进模型
Pascal VOC 2012 数据集 (20 类)
"""

import os
import sys
import gradio as gr
import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.detector import Detector
from utils.visualize import draw_detections, create_summary_text


# ============================================================
# 全局检测器实例（启动时加载一次）
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_weights.pt')

# 如果本地没有模型文件，尝试从 Hugging Face Hub 下载
HF_MODEL_REPO = "wanlixue/yolov8-cbam-voc2012"

if not os.path.exists(MODEL_PATH):
    print(f"[INFO] 本地模型文件不存在: {MODEL_PATH}")
    print(f"[INFO] 尝试从 Hugging Face Hub 下载: {HF_MODEL_REPO}")
    try:
        from huggingface_hub import hf_hub_download
        MODEL_PATH = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="model_weights.pt",
            local_dir=os.path.dirname(__file__),
        )
        print(f"[INFO] 模型下载成功: {MODEL_PATH}")
    except ImportError:
        print("[WARN] huggingface_hub 未安装，无法自动下载模型")
        print("[WARN] 请手动放置 model_weights.pt 到项目根目录")
    except Exception as e:
        print(f"[WARN] 模型下载失败: {e}")
        print("[WARN] 请手动放置 model_weights.pt 到项目根目录")

print("=" * 60)
print("目标检测系统启动中...")
print("模型: YOLOv8 + CBAM (Pascal VOC 2012)")
print("=" * 60)

detector = Detector(weights_path=MODEL_PATH)


# ============================================================
# 检测函数
# ============================================================
def detect_objects(image, conf_thres=0.25, iou_thres=0.45):
    """
    目标检测核心函数

    Args:
        image: numpy array (H, W, 3) from Gradio
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值

    Returns:
        annotated_image: 绘制了检测框的图片
        summary: 检测结果摘要
    """
    if image is None:
        return None, "请上传一张图片。"

    # Gradio 输入是 RGB，Detector 内部处理 BGR 转换
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 运行检测
    detections = detector.detect(image_bgr, conf_thres=conf_thres, iou_thres=iou_thres)

    # 绘制检测结果
    annotated = draw_detections(
        image,  # RGB
        detections,
        detector.class_names,
        detector.colors
    )

    # 生成摘要
    summary = create_summary_text(detections, detector.class_names)

    return annotated, summary


# ============================================================
# 构建 Gradio 界面
# ============================================================
TITLE = """
# 🔍 目标检测系统

基于 YOLOv8 改进模型，集成 **CBAM 注意力机制**、**Transformer 自注意力** 和 **Mamba 状态空间模型**
"""

DESCRIPTION = """
上传一张图片，系统将自动检测其中的 **20 类目标**：

<div style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: center;">
    <span style="background: #e3f2fd; padding: 4px 12px; border-radius: 12px;">✈️ 飞机</span>
    <span style="background: #fce4ec; padding: 4px 12px; border-radius: 12px;">🚲 自行车</span>
    <span style="background: #e8f5e9; padding: 4px 12px; border-radius: 12px;">🐦 鸟</span>
    <span style="background: #fff3e0; padding: 4px 12px; border-radius: 12px;">🚤 船</span>
    <span style="background: #f3e5f5; padding: 4px 12px; border-radius: 12px;">🍶 瓶子</span>
    <span style="background: #e0f7fa; padding: 4px 12px; border-radius: 12px;">🚌 公交车</span>
    <span style="background: #fff9c4; padding: 4px 12px; border-radius: 12px;">🚗 汽车</span>
    <span style="background: #f1f8e9; padding: 4px 12px; border-radius: 12px;">🐱 猫</span>
    <span style="background: #ede7f6; padding: 4px 12px; border-radius: 12px;">🪑 椅子</span>
    <span style="background: #fbe9e7; padding: 4px 12px; border-radius: 12px;">🐄 牛</span>
    <span style="background: #eceff1; padding: 4px 12px; border-radius: 12px;">🍽️ 餐桌</span>
    <span style="background: #fce4ec; padding: 4px 12px; border-radius: 12px;">🐶 狗</span>
    <span style="background: #e8eaf6; padding: 4px 12px; border-radius: 12px;">🐴 马</span>
    <span style="background: #e0f2f1; padding: 4px 12px; border-radius: 12px;">🏍️ 摩托车</span>
    <span style="background: #fff8e1; padding: 4px 12px; border-radius: 12px;">🧑 人</span>
    <span style="background: #f1f8e9; padding: 4px 12px; border-radius: 12px;">🪴 盆栽</span>
    <span style="background: #f9fbe7; padding: 4px 12px; border-radius: 12px;">🐑 羊</span>
    <span style="background: #efebe9; padding: 4px 12px; border-radius: 12px;">🛋️ 沙发</span>
    <span style="background: #e1f5fe; padding: 4px 12px; border-radius: 12px;">🚆 火车</span>
    <span style="background: #f3e5f5; padding: 4px 12px; border-radius: 12px;">📺 显示器</span>
</div>
"""

ARTICLE = """
---

## 📖 关于本项目

本系统基于 **YOLOv8** 目标检测架构，并在以下方面进行了改进：

- 🧠 **CBAM 注意力机制** — 增强特征表达，提升检测精度
- 🔄 **Transformer 自注意力** — 捕捉全局上下文信息
- ⚡ **Mamba 状态空间模型** — 高效的长序列建模

**训练数据集**: Pascal VOC 2012

**技术栈**: PyTorch | Gradio | OpenCV

---

## 🔗 链接

- [GitHub 仓库]()
- [Hugging Face Spaces]()
"""


def create_ui():
    """创建 Gradio 界面"""
    with gr.Blocks(
        title="目标检测系统 - YOLOv8",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Noto Sans SC"), "Arial", "sans-serif"],
        ),
        css="""
        .output-image img {
            max-height: 600px;
            object-fit: contain;
        }
        footer { visibility: hidden }
        """,
    ) as demo:
        gr.Markdown(TITLE)

        with gr.Row():
            with gr.Column(scale=3):
                # 输入区域
                input_image = gr.Image(
                    label="📤 上传图片",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=400,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        conf_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.25,
                            step=0.05,
                            label="🎯 置信度阈值",
                            info="值越低检测越多（可能误检），值越高检测越准（可能漏检）",
                        )
                    with gr.Column(scale=1):
                        iou_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="📐 NMS IoU 阈值",
                            info="重叠框的过滤强度",
                        )

                submit_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

            with gr.Column(scale=4):
                # 输出区域
                output_image = gr.Image(
                    label="📊 检测结果",
                    type="numpy",
                    height=400,
                    elem_classes="output-image",
                )
                output_summary = gr.Textbox(
                    label="📋 检测摘要",
                    lines=6,
                    placeholder="检测结果将显示在这里...",
                )

        # 示例图片区域
        gr.Markdown("### 📸 示例图片（点击加载）")
        gr.Examples(
            examples=[
                # 如果有示例图片就加上，否则留空
            ],
            inputs=input_image,
            label="",
        )

        # 绑定检测函数
        submit_btn.click(
            fn=detect_objects,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

        # 上传图片后自动检测
        input_image.change(
            fn=detect_objects,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

        # 页脚
        gr.Markdown(ARTICLE)

    return demo


# ============================================================
# 启动应用
# ============================================================
if __name__ == '__main__':
    demo = create_ui()
    demo.queue(max_size=20, default_concurrency_limit=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )