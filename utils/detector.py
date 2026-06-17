"""
目标检测器模块
基于 ultralytics YOLOv8 预训练模型 (COCO 80类)
"""

import numpy as np
import cv2
from ultralytics import YOLO
import os


# COCO 数据集 80 类中文名称映射
COCO_CLASSES_ZH = {
    0: '人', 1: '自行车', 2: '汽车', 3: '摩托车', 4: '飞机',
    5: '公交车', 6: '火车', 7: '卡车', 8: '船', 9: '红绿灯',
    10: '消火栓', 11: '停车标志', 12: '停车计时器', 13: '长椅', 14: '鸟',
    15: '猫', 16: '狗', 17: '马', 18: '羊', 19: '牛',
    20: '大象', 21: '熊', 22: '斑马', 23: '长颈鹿', 24: '背包',
    25: '雨伞', 26: '手提包', 27: '领带', 28: '行李箱', 29: '飞盘',
    30: '滑雪板', 31: '雪板', 32: '球', 33: '风筝', 34: '棒球棒',
    35: '手套', 36: '滑板', 37: '冲浪板', 38: '网球拍', 39: '瓶子',
    40: '酒杯', 41: '杯子', 42: '叉子', 43: '刀', 44: '勺子',
    45: '碗', 46: '香蕉', 47: '苹果', 48: '三明治', 49: '橙子',
    50: '西兰花', 51: '胡萝卜', 52: '热狗', 53: '披萨', 54: '甜甜圈',
    55: '蛋糕', 56: '椅子', 57: '沙发', 58: '盆栽', 59: '床',
    60: '餐桌', 61: '马桶', 62: '显示器', 63: '笔记本', 64: '鼠标',
    65: '遥控器', 66: '键盘', 67: '手机', 68: '微波炉', 69: '烤箱',
    70: '烤面包机', 71: '水槽', 72: '冰箱', 73: '书', 74: '钟',
    75: '花瓶', 76: '剪刀', 77: '泰迪熊', 78: '吹风机', 79: '牙刷'
}

COCO_CLASSES_EN = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


class Detector:
    """YOLOv8 目标检测器（使用 ultralytics 预训练模型）"""

    def __init__(self, model_name='yolov8n.pt', device=None):
        """
        初始化检测器

        Args:
            model_name: 模型名称或路径
                - 'yolov8n.pt'  Nano (最快, ~6MB)
                - 'yolov8s.pt'  Small (~22MB)
                - 'yolov8m.pt'  Medium (~52MB)
            device: 推理设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.device = device or ('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        # 加载模型
        # 优先使用本地模型文件，否则尝试从镜像下载
        local_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', model_name)
        if os.path.exists(local_model):
            model_name = local_model
        elif os.path.exists(model_name):
            pass  # 使用用户指定的路径
        else:
            # 尝试从 Hugging Face 镜像下载
            print(f"[INFO] 本地模型不存在，尝试从镜像下载: {model_name}")
            try:
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                from huggingface_hub import hf_hub_download
                model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                os.makedirs(model_dir, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id='Ultralytics/YOLOv8',
                    filename=model_name,
                    local_dir=model_dir,
                )
                model_name = downloaded
                print(f"[INFO] 模型下载成功: {model_name}")
            except Exception as e:
                print(f"[WARN] 镜像下载失败: {e}，将尝试 ultralytics 默认下载")

        self.model_path = model_name
        self.model = YOLO(model_name)

        # 类别信息
        self.class_names_en = COCO_CLASSES_EN
        self.class_names_zh = COCO_CLASSES_ZH
        self.class_names = [f"{COCO_CLASSES_EN[i]} ({COCO_CLASSES_ZH[i]})" for i in range(80)]
        self.num_classes = 80
        self.class_names_en_only = [COCO_CLASSES_EN[i] for i in range(80)]

        # 为每个类别生成颜色
        np.random.seed(42)
        self.colors = np.random.randint(50, 255, size=(self.num_classes, 3), dtype=np.uint8).tolist()

        print(f"[INFO] 模型加载完成: {model_name}")
        print(f"[INFO] 类别数量: {self.num_classes} (COCO)")

    def detect(self, image, conf_thres=0.25, iou_thres=0.45, max_det=100):
        """
        对图片进行目标检测

        Args:
            image: numpy array (H, W, 3) BGR 图片
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值
            max_det: 最大检测数量

        Returns:
            detections: list of [x1, y1, x2, y2, score, class_id]
        """
        # 转换为 RGB（ultralytics 需要 RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 运行检测
        results = self.model(
            image_rgb,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            verbose=False,
        )

        # 解析结果
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()       # [N, 4] x1,y1,x2,y2
                scores = result.boxes.conf.cpu().numpy()      # [N]
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append([x1, y1, x2, y2, float(score), int(cls_id)])

                # 按置信度降序
                detections.sort(key=lambda x: x[4], reverse=True)

        return detections

    def get_class_name(self, class_id, lang='zh'):
        """获取类别名称"""
        if lang == 'zh':
            return self.class_names_zh.get(class_id, f"未知_{class_id}")
        return self.class_names_en.get(class_id, f"unknown_{class_id}")

    def get_color(self, class_id):
        """获取类别对应颜色"""
        if 0 <= class_id < len(self.colors):
            return tuple(self.colors[class_id])
        return (255, 255, 255)
