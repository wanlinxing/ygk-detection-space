"""
检测结果可视化模块
在图片上绘制检测框、类别标签和置信度
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO


def draw_detections_cv2(image, detections, class_names, colors):
    """
    使用 OpenCV 绘制检测结果（适合中文环境用 PIL 方案更好）

    Args:
        image: numpy array (H, W, 3) BGR
        detections: list of [x1, y1, x2, y2, score, class_id]
        class_names: list of class name strings
        colors: list of (R, G, B) tuples

    Returns:
        image with boxes drawn (BGR)
    """
    result = image.copy()

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)

        color = tuple(colors[cls_id]) if cls_id < len(colors) else (0, 255, 0)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"{class_name}: {score:.2f}"

        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # 绘制标签文字
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def draw_detections_pil(image, detections, class_names, colors, font_path=None):
    """
    使用 PIL 绘制检测结果（支持中文，效果更好）

    Args:
        image: numpy array (H, W, 3) RGB
        detections: list of [x1, y1, x2, y2, score, class_id]
        class_names: list of class name strings
        colors: list of (R, G, B) tuples
        font_path: 中文字体路径（可选）

    Returns:
        image with boxes drawn (RGB, numpy array)
    """
    result = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(result)

    # 尝试加载字体
    try:
        if font_path:
            font = ImageFont.truetype(font_path, 16)
            small_font = ImageFont.truetype(font_path, 12)
        else:
            # 尝试系统默认字体
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)

        color = tuple(colors[cls_id]) if cls_id < len(colors) else (0, 255, 0)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"{class_name}: {score:.2f}"

        # 绘制边界框（加粗）
        for offset in range(2):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
                width=1
            )

        # 绘制标签背景和文字
        try:
            text_bbox = draw.textbbox((0, 0), label, font=small_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = 60, 14

        label_y = max(0, y1 - text_h - 6)
        draw.rectangle([x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255), font=small_font)

    return np.array(result)


def draw_detections(image, detections, class_names, colors):
    """
    综合绘制检测结果（PIL 方案，RGB 输入）
    """
    return draw_detections_pil(image, detections, class_names, colors)


def create_summary_text(detections, class_names):
    """
    生成检测结果摘要文本

    Args:
        detections: list of [x1, y1, x2, y2, score, class_id]
        class_names: list of class name strings

    Returns:
        summary string
    """
    if not detections:
        return "未检测到任何目标。"

    # 统计各类别数量
    class_counts = {}
    for det in detections:
        cls_id = int(det[5])
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"未知_{cls_id}"
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    lines = [f"共检测到 {len(detections)} 个目标:"]
    for name, count in class_counts.items():
        lines.append(f"  • {name}: {count} 个")

    return "\n".join(lines)
