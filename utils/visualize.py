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


def draw_detections_pil_enhanced(image, detections, class_names, colors):
    """
    增强版 PIL 绘制，支持阴影、圆角、半透明等美化效果

    Args:
        image: numpy array (H, W, 3) RGB
        detections: list of [x1, y1, x2, y2, score, class_id]
        class_names: list of class name strings
        colors: list of (R, G, B) tuples

    Returns:
        image with beautified boxes drawn (RGB, numpy array)
    """
    from PIL import ImageDraw

    h, w = image.shape[:2]
    result = Image.fromarray(image.astype(np.uint8))
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(result)

    # 尝试加载字体
    try:
        # Windows 中文字体
        for font_path in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyhbd.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]:
            try:
                font = ImageFont.truetype(font_path, 16)
                small_font = ImageFont.truetype(font_path, 12)
                break
            except Exception:
                continue
        else:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # 按置信度降序绘制（高置信度在上层）
    sorted_dets = sorted(detections, key=lambda x: x[4])

    for det in sorted_dets:
        x1, y1, x2, y2, score, cls_id = det
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        cls_id = int(cls_id)
        if x2 <= x1 or y2 <= y1:
            continue

        color = tuple(colors[cls_id]) if cls_id < len(colors) else (0, 255, 0)
        class_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

        # 根据置信度调整线宽
        if score >= 0.7:
            line_width = 4
            shadow_offset = 3
        elif score >= 0.4:
            line_width = 3
            shadow_offset = 2
        else:
            line_width = 2
            shadow_offset = 1

        # === 绘制阴影 ===
        shadow_color = (0, 0, 0, 40)
        for dx in range(shadow_offset):
            overlay_draw.rectangle(
                [x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset],
                outline=shadow_color,
                width=line_width + 2,
            )

        # === 绘制半透明填充 ===
        fill_alpha = 25
        if score >= 0.7:
            fill_alpha = 35
        fill_rgba = color + (fill_alpha,)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_rgba)

        # === 绘制主边界框 ===
        # 外层（深色边缘）
        dark_color = tuple(max(0, c - 40) for c in color)
        overlay_draw.rectangle([x1, y1, x2, y2], outline=dark_color + (220,), width=line_width + 2)
        # 内层（亮色）
        overlay_draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=line_width)

        # === 绘制角标（四角加粗） ===
        corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1),
        ]:
            overlay_draw.line(
                [(cx, cy), (cx + corner_len * dx, cy), (cx, cy), (cx, cy + corner_len * dy)],
                fill=color + (230,),
                width=line_width + 1,
            )

        # === 绘制标签 ===
        label_text = f"{class_name} {score:.0%}"
        try:
            text_bbox = draw.textbbox((0, 0), label_text, font=small_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = len(label_text) * 7, 14

        padding = 6
        label_bg_w = text_w + padding * 2
        label_bg_h = text_h + padding

        # 标签位置（优先放框内顶部，框太小就放框外）
        if y1 > label_bg_h + 8:
            label_y = y1 - label_bg_h - 2
            label_bg_y1 = label_y
            label_bg_y2 = y1
        else:
            label_y = y1 + 2
            label_bg_y1 = y1
            label_bg_y2 = y1 + label_bg_h

        label_x = x1 + 2

        # 标签背景（扩展到框外）
        overlay_draw.rectangle(
            [label_x, label_bg_y1, label_x + label_bg_w, label_bg_y2],
            fill=color + (200,),
        )

        # 标签文字
        draw.text(
            (label_x + padding, label_y + padding // 2),
            label_text,
            fill=(255, 255, 255),
            font=small_font,
        )

    # 合成
    result = Image.alpha_composite(result.convert('RGBA'), overlay)
    return np.array(result.convert('RGB'))


def draw_detections(image, detections, class_names, colors):
    """
    综合绘制检测结果
    """
    try:
        return draw_detections_pil_enhanced(image, detections, class_names, colors)
    except Exception:
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
