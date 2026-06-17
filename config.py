"""
目标检测项目配置文件
使用YOLOv8进行目标检测，基于PyTorch实现
数据集：Pascal VOC 2012
"""

import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集配置
DATASET_CONFIG = {
    'name': 'VOC2012',
    'num_classes': 20,
    'classes': [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ],
    'data_dir': os.path.join(ROOT_DIR, 'data', 'VOCdevkit', 'VOC2012'),
    'image_size': 416,
    'train_val_split': 0.8,
}

# 模型配置
MODEL_CONFIG = {
    'name': 'YOLOv8',
    'backbone': 'CSPDarknet',
    'neck': 'PANet',
    'head': 'YOLOHead',
    'anchors': [
        [10, 13, 16, 30, 33, 23],      # P3/8
        [30, 61, 62, 45, 59, 119],     # P4/16
        [116, 90, 156, 198, 373, 326]  # P5/32
    ],
    'num_classes': 20,
    'input_size': 416,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 1,  # 显存不足，使用最小batch_size
    'epochs': 10,  # 减少epoch数量以加快训练
    'lr': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'save_dir': os.path.join(ROOT_DIR, 'runs', 'train'),
    'device': 'cuda',  # 如果有GPU则使用cuda，否则使用cpu
}

# 数据增强配置
AUGMENT_CONFIG = {
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,    # HSV-Saturation augmentation
    'hsv_v': 0.4,    # HSV-Value augmentation
    'degrees': 0.0,  # Rotation
    'translate': 0.1,  # Translation
    'scale': 0.5,    # Scale
    'shear': 0.0,    # Shear
    'flipud': 0.0,   # Flip up-down
    'fliplr': 0.5,   # Flip left-right
    'mosaic': 1.0,   # Mosaic augmentation
    'mixup': 0.0,    # Mixup augmentation
}

# 评估配置
VAL_CONFIG = {
    'batch_size': 4,  # 验证batch_size也减小
    'conf_thres': 0.001,
    'iou_thres': 0.6,
    'max_det': 300,
    'save_dir': os.path.join(ROOT_DIR, 'runs', 'val'),
}

# 测试配置
TEST_CONFIG = {
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'max_det': 1000,
    'save_dir': os.path.join(ROOT_DIR, 'runs', 'test'),
}
