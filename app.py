"""
目标检测 Web 应用 - Gradio 界面
基于 YOLOv8 预训练模型 (COCO 80类)
"""

import os
import sys
import gradio as gr
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import Detector
from visualize import draw_detections as draw_boxes

# ============================================================
# 全局配置
# ============================================================
MODEL_NAME = 'yolov8s.pt'

print("=" * 60)
print("智能目标检测系统启动中...")
print(f"模型: YOLOv8 Small (COCO 80类)")
print("=" * 60)

detector = Detector(model_name=MODEL_NAME)


# ============================================================
# 样式定义
# ============================================================
CUSTOM_CSS = """
/* === 全局样式 === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f0 100%) !important;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important;
}

/* === 主容器 === */
.main-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
    padding: 40px;
    margin: 20px;
    border: 1px solid rgba(255, 255, 255, 0.6);
}

/* === 标题区域 === */
.header-section {
    text-align: center;
    margin-bottom: 36px;
    position: relative;
}

.header-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 6px 20px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.header-title {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 12px 0;
    letter-spacing: -0.5px;
}

.header-subtitle {
    font-size: 16px;
    color: #64748b;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* === 状态指示器 === */
.status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    margin-bottom: 32px;
    flex-wrap: wrap;
}

.status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: white;
    padding: 10px 20px;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    font-size: 14px;
    font-weight: 500;
    color: #334155;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

.status-dot.warn {
    background: #f59e0b;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
}

/* === 卡片容器 === */
.card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.06);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-title .icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.icon-upload { background: linear-gradient(135deg, #dbeafe, #bfdbfe); }
.icon-sliders { background: linear-gradient(135deg, #fef3c7, #fde68a); }
.icon-result { background: linear-gradient(135deg, #dcfce7, #bbf7d0); }
.icon-stats { background: linear-gradient(135deg, #fce7f3, #fbcfe8); }

/* === 按钮样式 === */
.detect-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 32px !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
}

.detect-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    background: linear-gradient(135deg, #5a6fd6 0%, #6a3f9e 100%) !important;
}

.detect-btn:active {
    transform: translateY(0) !important;
}

/* === 滑块样式 === */
input[type="range"] {
    accent-color: #667eea !important;
    height: 6px !important;
}

/* === 图片区域 === */
.image-container {
    border-radius: 16px;
    overflow: hidden;
    border: 2px dashed #e2e8f0;
    transition: all 0.3s ease;
    background: #f8fafc;
}

.image-container:has(img) {
    border: 2px solid #e2e8f0;
}

.image-container:hover {
    border-color: #667eea;
}

/* === 检测摘要卡片 === */
.detection-summary {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    min-height: 80px;
}

.detection-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: white;
    padding: 6px 14px;
    border-radius: 50px;
    margin: 4px;
    font-size: 13px;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.detection-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.detection-count {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 2px 10px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 12px;
}

.no-detection {
    text-align: center;
    color: #94a3b8;
    font-size: 15px;
    padding: 24px;
}

.no-detection .icon-large {
    font-size: 48px;
    display: block;
    margin-bottom: 12px;
    opacity: 0.5;
}

/* === 统计面板 === */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 12px;
}

.stat-box {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
    transition: all 0.2s ease;
}

.stat-box:hover {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    border-color: #c7d2fe;
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* === 置信度颜色 === */
.conf-high { color: #22c55e; font-weight: 700; }
.conf-medium { color: #f59e0b; font-weight: 700; }
.conf-low { color: #ef4444; font-weight: 700; }

/* === 页脚 === */
.footer {
    text-align: center;
    margin-top: 48px;
    padding: 28px;
    color: #94a3b8;
    font-size: 13px;
    border-top: 1px solid #e2e8f0;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 12px;
}

.footer-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s;
}

.footer-link:hover {
    color: #764ba2;
}

/* === 响应式 === */
@media (max-width: 768px) {
    .header-title { font-size: 28px; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

# ============================================================
# HTML 组件
# ============================================================
HEADER_HTML = """
<div class="header-section">
    <div class="header-badge">🔬 Deep Learning · Object Detection</div>
    <h1 class="header-title">智能目标检测系统</h1>
    <p class="header-subtitle">
        基于 <strong>YOLOv8</strong> 深度学习模型，支持 80 类日常物体实时检测
        —— 上传图片，即刻识别
    </p>
    <div class="status-bar">
        <div class="status-item">
            <span class="status-dot"></span>
            <span>YOLOv8 Small · 已就绪</span>
        </div>
        <div class="status-item">
            <span>🎯</span>
            <span>80 类目标</span>
        </div>
        <div class="status-item">
            <span>⚡</span>
            <span>实时推理</span>
        </div>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="footer">
    <div class="footer-links">
        <a class="footer-link" href="https://docs.ultralytics.com/" target="_blank">📚 YOLOv8 文档</a>
        <a class="footer-link" href="https://cocodataset.org/" target="_blank">📊 COCO 数据集</a>
        <a class="footer-link" href="https://github.com/ultralytics/ultralytics" target="_blank">💻 GitHub</a>
    </div>
    <p>© 2026 深度学习期末综合设计 · Powered by PyTorch & Gradio</p>
</div>
"""


# ============================================================
# 辅助函数
# ============================================================
def get_confidence_class(score):
    """根据置信度返回 CSS 类名"""
    if score >= 0.7:
        return "conf-high"
    elif score >= 0.4:
        return "conf-medium"
    else:
        return "conf-low"


def build_summary_html(detections, class_names_en, class_names_zh):
    """构建检测摘要 HTML"""
    if not detections:
        return """
        <div class="no-detection">
            <span class="icon-large">🔍</span>
            <p>未检测到任何目标</p>
            <p style="font-size:13px;color:#94a3b8;">尝试降低置信度阈值或上传其他图片</p>
        </div>
        """

    # 统计信息
    class_counts = {}
    for det in detections:
        cls_id = int(det[5])
        en_name = class_names_en.get(cls_id, f"cls_{cls_id}")
        zh_name = class_names_zh.get(cls_id, f"未知")
        key = (en_name, zh_name)
        class_counts[key] = class_counts.get(key, 0) + 1

    total = len(detections)
    unique_classes = len(class_counts)

    # 构建 HTML
    html = '<div style="margin-bottom:20px;">'

    # 统计面板
    html += '<div class="stats-grid" style="margin-bottom:20px;">'
    html += f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">检测目标</div></div>'
    html += f'<div class="stat-box"><div class="stat-number">{unique_classes}</div><div class="stat-label">类别数</div></div>'
    top_score = max(d[4] for d in detections) if detections else 0
    html += f'<div class="stat-box"><div class="stat-number">{top_score:.0%}</div><div class="stat-label">最高置信度</div></div>'
    avg_score = sum(d[4] for d in detections) / total if total > 0 else 0
    html += f'<div class="stat-box"><div class="stat-number">{avg_score:.0%}</div><div class="stat-label">平均置信度</div></div>'
    html += '</div>'

    # 检测列表
    html += '<div style="font-size:14px;font-weight:600;color:#475569;margin-bottom:10px;">📋 检测详情</div>'
    html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">'
    for (en_name, zh_name), count in sorted(class_counts.items(), key=lambda x: -x[1]):
        html += (
            f'<div class="detection-item">'
            f'<span>{zh_name}</span>'
            f'<span style="color:#94a3b8;font-size:11px;">{en_name}</span>'
            f'<span class="detection-count">×{count}</span>'
            f'</div>'
        )
    html += '</div>'

    # 置信度分布
    html += '<div style="margin-top:16px;font-size:12px;color:#94a3b8;">'
    high_conf = sum(1 for d in detections if d[4] >= 0.7)
    mid_conf = sum(1 for d in detections if 0.4 <= d[4] < 0.7)
    low_conf = sum(1 for d in detections if d[4] < 0.4)
    html += f'🟢 高置信度 (≥70%): {high_conf} 个 · '
    html += f'🟡 中置信度 (40-70%): {mid_conf} 个 · '
    html += f'🔴 低置信度 (&lt;40%): {low_conf} 个'
    html += '</div>'

    html += '</div>'

    return html


def build_empty_summary_html():
    """空检测的 HTML 占位"""
    return """
    <div class="no-detection">
        <span class="icon-large">📸</span>
        <p>等待上传图片...</p>
        <p style="font-size:13px;color:#94a3b8;">上传图片后自动检测，或点击"开始检测"</p>
    </div>
    """


# ============================================================
# 检测核心函数
# ============================================================
def detect_and_render(image, conf_thres=0.25, iou_thres=0.45):
    """检测 + 渲染可视化结果"""
    if image is None:
        return None, build_empty_summary_html()

    # BGR 转换
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 检测
    detections = detector.detect(image_bgr, conf_thres=conf_thres, iou_thres=iou_thres)

    # 绘制边界框
    annotated = draw_boxes(image, detections, detector.class_names, detector.colors)

    # 构建 HTML 摘要
    summary_html = build_summary_html(
        detections,
        detector.class_names_en,
        detector.class_names_zh
    )

    return annotated, summary_html


# ============================================================
# 构建界面
# ============================================================
def create_ui():
    with gr.Blocks(title="智能目标检测系统 · YOLOv8") as demo:

        # 顶部 Header
        gr.HTML(HEADER_HTML)

        # 主体布局
        with gr.Row(equal_height=True):
            # === 左侧：上传区域 ===
            with gr.Column(scale=5, min_width=320):
                with gr.Group(elem_classes="card"):
                    gr.HTML('<div class="card-title"><span class="icon icon-upload">📤</span> 上传图片</div>')

                    input_image = gr.Image(
                        label=None,
                        type="numpy",
                        sources=["upload", "clipboard", "webcam"],
                        height=420,
                        elem_classes="image-container",
                        show_label=False,
                    )

                    with gr.Row():
                        conf_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.95,
                            value=0.25,
                            step=0.05,
                            label="🎯 置信度阈值",
                            info="值越低检测越多（可能误检），越高越精准",
                        )
                        iou_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="📐 IoU 阈值",
                            info="重叠框过滤强度，越低去重越强",
                        )

                    detect_btn = gr.Button(
                        "🚀 开始检测",
                        variant="primary",
                        size="lg",
                        elem_classes="detect-btn",
                    )

            # === 右侧：结果区域 ===
            with gr.Column(scale=5, min_width=320):
                with gr.Group(elem_classes="card"):
                    gr.HTML('<div class="card-title"><span class="icon icon-result">📊</span> 检测结果</div>')
                    output_image = gr.Image(
                        label=None,
                        type="numpy",
                        height=420,
                        elem_classes="image-container",
                        show_label=False,
                    )

                with gr.Group(elem_classes="card"):
                    gr.HTML('<div class="card-title"><span class="icon icon-stats">📋</span> 检测摘要</div>')
                    output_summary = gr.HTML(
                        value=build_empty_summary_html(),
                        elem_classes="detection-summary",
                    )

        # 页脚
        gr.HTML(FOOTER_HTML)

        # === 事件绑定 ===
        detect_btn.click(
            fn=detect_and_render,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

        input_image.change(
            fn=detect_and_render,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, output_summary],
        )

    return demo


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    demo = create_ui()
    demo.queue(max_size=20, default_concurrency_limit=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        head="""
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="基于 YOLOv8 的智能目标检测系统">
        """,
        favicon_path=None,
    )
