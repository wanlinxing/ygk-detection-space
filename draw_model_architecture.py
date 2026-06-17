"""
生成详细版模型结构图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# 颜色定义
COLOR_INPUT = '#005293'
COLOR_STEM = '#0077B6'
COLOR_STAGE = '#0096C7'
COLOR_CBAM = '#00B4D8'
COLOR_SPPF = '#48CAE4'
COLOR_NECK = '#90E0EF'
COLOR_TRANS = '#7209B7'
COLOR_HEAD = '#F72585'
COLOR_OUTPUT = '#4CC9F0'
COLOR_ARROW = '#00B4D8'
COLOR_TEXT = '#333333'
COLOR_BG = '#F0F8FF'

# 绘制带圆角的模块框
def draw_module(ax, x, y, w, h, color, text, fontsize=10, text_color='white', alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)
    return box

# 绘制箭头
def draw_arrow(ax, start, end, color=COLOR_ARROW, style='->', lw=2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# 绘制虚线框（分组）
def draw_group_box(ax, x, y, w, h, color, label, fontsize=12):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                           facecolor='none', edgecolor=color, linewidth=2, linestyle='--', alpha=0.6)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h + 0.15, label, ha='center', va='bottom',
            fontsize=fontsize, color=color, fontweight='bold')

# ===================== 开始绘制 =====================

# 标题
ax.text(9, 13.5, '基于改进YOLOv8的目标检测模型结构', ha='center', va='center',
        fontsize=20, fontweight='bold', color=COLOR_INPUT)
ax.text(9, 13.1, '输入416×416 | Backbone+CSPDarknet+CBAM | Neck+PANet | Transformer | Detect Head',
        ha='center', va='center', fontsize=11, color='#666666')

# ====== 输入层 ======
draw_module(ax, 7.5, 12.0, 3, 0.7, COLOR_INPUT, '输入图像\n416×416×3', fontsize=11)
draw_arrow(ax, (9, 12.0), (9, 11.5))

# ====== Backbone 分组框 ======
draw_group_box(ax, 0.5, 5.5, 8.5, 5.8, COLOR_STAGE, 'Backbone (CSPDarknet + CBAM)', fontsize=13)

# Stem
draw_module(ax, 3.0, 10.5, 3, 0.7, COLOR_STEM, 'Stem\nConv 3→64, stride=2\n208×208×64', fontsize=9)
draw_arrow(ax, (4.5, 10.5), (4.5, 10.0))

# Stage1 P2/4
draw_module(ax, 3.0, 9.3, 3, 0.7, COLOR_STAGE, 'Stage1 (P2/4)\nConv + C2f×3\n104×104×128', fontsize=9)
draw_arrow(ax, (4.5, 9.3), (4.5, 8.8))

# Stage2 P3/8
draw_module(ax, 3.0, 8.1, 3, 0.7, COLOR_STAGE, 'Stage2 (P3/8)\nConv + C2f×6\n52×52×256', fontsize=9)
draw_arrow(ax, (4.5, 8.1), (4.5, 7.6))

# CBAM-P3
draw_module(ax, 3.0, 6.9, 3, 0.7, COLOR_CBAM, 'CBAM-P3\n通道+空间注意力\n52×52×256', fontsize=9)
# 从CBAM-P3引出到Neck的线
draw_arrow(ax, (6.0, 7.25), (8.0, 7.25), style='->', lw=1.5)
ax.text(7.0, 7.45, 'c3', ha='center', va='bottom', fontsize=8, color=COLOR_CBAM, fontweight='bold')
draw_arrow(ax, (4.5, 6.9), (4.5, 6.4))

# Stage3 P4/16
draw_module(ax, 3.0, 5.7, 3, 0.7, COLOR_STAGE, 'Stage3 (P4/16)\nConv + C2f×6\n26×26×512', fontsize=9)
draw_arrow(ax, (4.5, 5.7), (4.5, 5.2))

# CBAM-P4
draw_module(ax, 3.0, 4.5, 3, 0.7, COLOR_CBAM, 'CBAM-P4\n通道+空间注意力\n26×26×512', fontsize=9)
draw_arrow(ax, (6.0, 4.85), (8.0, 4.85), style='->', lw=1.5)
ax.text(7.0, 5.05, 'c4', ha='center', va='bottom', fontsize=8, color=COLOR_CBAM, fontweight='bold')
draw_arrow(ax, (4.5, 4.5), (4.5, 4.0))

# Stage4 P5/32 + SPPF
draw_module(ax, 3.0, 3.3, 3, 0.7, COLOR_SPPF, 'Stage4 (P5/32)\nConv + C2f×3 + SPPF\n13×13×1024', fontsize=9)
draw_arrow(ax, (4.5, 3.3), (4.5, 2.8))

# CBAM-P5
draw_module(ax, 3.0, 2.1, 3, 0.7, COLOR_CBAM, 'CBAM-P5\n通道+空间注意力\n13×13×1024', fontsize=9)
draw_arrow(ax, (6.0, 2.45), (8.0, 2.45), style='->', lw=1.5)
ax.text(7.0, 2.65, 'c5', ha='center', va='bottom', fontsize=8, color=COLOR_CBAM, fontweight='bold')

# ====== Neck 分组框 ======
draw_group_box(ax, 8.0, 4.8, 4.0, 5.5, COLOR_NECK, 'Neck (PANet)', fontsize=13)

# 上采样 P5->P4
ax.annotate('', xy=(10.0, 4.85), xytext=(10.0, 2.45),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5, connectionstyle="arc3,rad=0.3"))
ax.text(10.3, 3.6, '上采样\n×2', ha='left', va='center', fontsize=8, color=COLOR_TEXT)

# Concat + C2f P4
draw_module(ax, 8.5, 4.3, 3, 0.55, COLOR_NECK, 'Concat + C2f\n26×26×512', fontsize=9, text_color=COLOR_TEXT)
draw_arrow(ax, (10.0, 4.3), (10.0, 3.8))

# 上采样 P4->P3
ax.annotate('', xy=(10.0, 6.95), xytext=(10.0, 3.8),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5, connectionstyle="arc3,rad=-0.3"))
ax.text(10.3, 5.3, '上采样\n×2', ha='left', va='center', fontsize=8, color=COLOR_TEXT)

# Concat + C2f P3
draw_module(ax, 8.5, 6.4, 3, 0.55, COLOR_NECK, 'Concat + C2f\n52×52×256', fontsize=9, text_color=COLOR_TEXT)
draw_arrow(ax, (10.0, 6.4), (10.0, 5.9))

# 下采样 P3->P4
draw_arrow(ax, (10.0, 5.9), (10.0, 5.4))
ax.text(10.3, 5.65, 'Conv\nstride=2', ha='left', va='center', fontsize=7, color=COLOR_TEXT)

# Concat + C2f P4_out
draw_module(ax, 8.5, 4.85, 3, 0.55, COLOR_NECK, 'Concat + C2f\n26×26×512', fontsize=9, text_color=COLOR_TEXT)
draw_arrow(ax, (10.0, 4.85), (10.0, 4.3))

# 下采样 P4->P5
ax.annotate('', xy=(10.0, 2.45), xytext=(10.0, 4.3),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=1.5, connectionstyle="arc3,rad=0.3"))
ax.text(10.3, 3.3, 'Conv\nstride=2', ha='left', va='center', fontsize=7, color=COLOR_TEXT)

# Concat + C2f P5_out
draw_module(ax, 8.5, 1.9, 3, 0.55, COLOR_NECK, 'Concat + C2f\n13×13×1024', fontsize=9, text_color=COLOR_TEXT)

# ====== Transformer Block ======
draw_group_box(ax, 12.5, 4.0, 3.0, 4.5, COLOR_TRANS, 'Transformer Block', fontsize=13)

draw_module(ax, 12.8, 6.4, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MSA(8h)\n52×52×256', fontsize=9)
draw_arrow(ax, (11.5, 6.65), (12.8, 6.65), style='->', lw=1.5)
draw_arrow(ax, (14.0, 6.4), (14.0, 5.9))

draw_module(ax, 12.8, 5.35, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MLP\n52×52×256', fontsize=9)
draw_arrow(ax, (14.0, 5.35), (14.0, 4.85))

draw_module(ax, 12.8, 4.3, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MSA(8h)\n26×26×512', fontsize=9)
draw_arrow(ax, (11.5, 4.55), (12.8, 4.55), style='->', lw=1.5)
draw_arrow(ax, (14.0, 4.3), (14.0, 3.8))

draw_module(ax, 12.8, 3.25, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MLP\n26×26×512', fontsize=9)
draw_arrow(ax, (14.0, 3.25), (14.0, 2.75))

draw_module(ax, 12.8, 2.2, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MSA(8h)\n13×13×1024', fontsize=9)
draw_arrow(ax, (11.5, 2.45), (12.8, 2.45), style='->', lw=1.5)
draw_arrow(ax, (14.0, 2.2), (14.0, 1.7))

draw_module(ax, 12.8, 1.15, 2.4, 0.55, COLOR_TRANS, 'LayerNorm + MLP\n13×13×1024', fontsize=9)

# ====== Detect Head ======
draw_group_box(ax, 15.5, 1.0, 2.5, 7.0, COLOR_HEAD, 'Detect Head', fontsize=13)

draw_module(ax, 15.7, 6.4, 2.1, 0.55, COLOR_HEAD, 'Conv 1×1\nP3: 52×52×75', fontsize=9)
draw_arrow(ax, (14.0, 6.65), (15.7, 6.65), style='->', lw=1.5)

draw_module(ax, 15.7, 4.85, 2.1, 0.55, COLOR_HEAD, 'Conv 1×1\nP4: 26×26×75', fontsize=9)
draw_arrow(ax, (14.0, 4.55), (15.7, 4.55), style='->', lw=1.5)

draw_module(ax, 15.7, 1.15, 2.1, 0.55, COLOR_HEAD, 'Conv 1×1\nP5: 13×13×75', fontsize=9)
draw_arrow(ax, (14.0, 1.45), (15.7, 1.45), style='->', lw=1.5)

# ====== 输出 ======
draw_module(ax, 7.5, 0.2, 3, 0.6, COLOR_OUTPUT, '输出：边界框坐标 + 类别概率 + 置信度', fontsize=10)

# 从Detect Head到输出的箭头（示意）
ax.annotate('', xy=(9, 0.8), xytext=(16.75, 1.15),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2, connectionstyle="arc3,rad=-0.2"))
ax.annotate('', xy=(9, 0.8), xytext=(16.75, 4.85),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2, connectionstyle="arc3,rad=-0.1"))
ax.annotate('', xy=(9, 0.8), xytext=(16.75, 6.4),
            arrowprops=dict(arrowstyle='->', color=COLOR_ARROW, lw=2, connectionstyle="arc3,rad=0.1"))

# ====== 图例 ======
legend_items = [
    (COLOR_INPUT, '输入层'),
    (COLOR_STEM, 'Stem'),
    (COLOR_STAGE, 'Stage (C2f)'),
    (COLOR_CBAM, 'CBAM注意力'),
    (COLOR_SPPF, 'SPPF'),
    (COLOR_NECK, 'Neck (PANet)'),
    (COLOR_TRANS, 'Transformer'),
    (COLOR_HEAD, '检测头'),
]

for i, (color, label) in enumerate(legend_items):
    x_pos = 0.5 + i * 2.2
    rect = FancyBboxPatch((x_pos, 0.05), 0.4, 0.25, boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='white')
    ax.add_patch(rect)
    ax.text(x_pos + 0.5, 0.17, label, ha='left', va='center', fontsize=9, color=COLOR_TEXT)

# 保存
plt.tight_layout()
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_architecture_detailed.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] 详细版模型结构图已生成：{save_path}")
