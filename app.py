"""
目标检测 Web 应用 - Gradio 界面
基于 YOLOv8 预训练模型 (COCO 80类)
"""

import os
import sys
import gradio as gr
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.detector import Detector
from utils.visualize import draw_detections, create_summary_text


# ============================================================
# 全局配置
# ============================================================
# 可选模型: yolov8n.pt (nano/最快), yolov8s.pt (small), yolov8m.pt (medium)
MODEL_NAME = 'yolov8s.pt'  # 推荐 small，平衡速度与精度

print("=" * 60)
print("目标检测系统启动中...")
print(f"模型: YOLOv8 Small (COCO 80类)")
print("=" * 60)

# 加载检测器（首次运行自动下载模型权重 ~22MB）
detector = Detector(model_name=MODEL_NAME)


# ============================================================
# 检测函数
# ============================================================
def detect_objects(image, conf_thres=0.25, iou_thres=0.45):
    """
    目标检测核心函数

    Args:
        image: numpy array (H, W, 3) RGB
        conf_thres: 置信度阈值
        iou_thres: NMS IoU 阈值

    Returns:
        annotated_image: 绘制了检测框的图片
        summary: 检测结果摘要
    """
    if image is None:
        return None, "请上传一张图片。"

    # 转换为 BGR（detector 内部处理）
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
# 🔍 智能目标检测系统

基于 **YOLOv8** 预训练模型，可检测 **80 类** 日常物体
"""

DESCRIPTION = """
上传一张图片，系统将自动检测其中的人、车辆、动物、家具、食物等 **80 类目标**。

### 🎯 检测类别一览

| 类别 | 示例 |
|------|------|
| 🧑 **人物** | 人 |
| 🚗 **车辆** | 汽车、公交车、自行车、摩托车、飞机、船、火车、卡车 |
| 🐱 **动物** | 猫、狗、鸟、马、羊、牛、大象、熊、斑马、长颈鹿 |
| 🍕 **食物** | 披萨、蛋糕、三明治、苹果、橙子、香蕉、热狗、甜甜圈 |
| 🪑 **家具** | 椅子、沙发、床、餐桌、马桶 |
| 📱 **电子产品** | 手机、笔记本、显示器、键盘、鼠标、遥控器 |
| 🧴 **日常用品** | 瓶子、杯子、碗、背包、雨伞、剪刀、书、钟 |
"""

ARTICLE = """
---

## 📖 关于本项目

- **模型**: YOLOv8 Small（ultralytics 预训练）
- **数据集**: COCO 2017（80类）
- **框架**: PyTorch + Gradio

## 🔗 链接

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO 数据集](https://cocodataset.org/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
"""


def create_ui():
    """创建 Gradio 界面"""
    with gr.Blocks(title="智能目标检测系统 - YOLOv8") as demo:
        gr.Markdown(TITLE)

        with gr.Row():
            with gr.Column(scale=3):
                input_image = gr.Image(
                    label="📤 上传图片",
                    type="numpy",
                    sources=["upload", "clipboard", "webcam"],
                    height=420,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        conf_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.25,
                            step=0.05,
                            label="🎯 置信度阈值",
                            info="越低检测越多（可能误检），越高检测越准",
                        )
                    with gr.Column(scale=1):
                        iou_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="📐 NMS IoU 阈值",
                            info="重叠框过滤强度",
                        )

                submit_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

            with gr.Column(scale=4):
                output_image = gr.Image(
                    label="📊 检测结果",
                    type="numpy",
                    height=420,
                    elem_classes="output-image",
                )
                output_summary = gr.Textbox(
                    label="📋 检测摘要",
                    lines=6,
                    placeholder="检测结果将显示在这里...",
                )

        # 示例图片
        gr.Markdown("### 📸 示例")
        gr.Examples(
            examples=[
                "https://ultralytics.com/images/bus.jpg",
                "https://ultralytics.com/images/zidane.jpg",
            ],
            inputs=input_image,
            label="点击加载官方示例",
        )

        # 事件绑定
        submit_btn.click(
            fn=detect_objects,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

        input_image.change(
            fn=detect_objects,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

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
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
        ),
        css="""
        .output-image img {
            max-height: 600px;
            object-fit: contain;
        }
        footer { visibility: hidden }
        """,
    )
