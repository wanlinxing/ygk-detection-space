"""
目标检测器模块
加载训练好的 YOLOv8 模型，提供图片推理和结果解码功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_CONFIG, MODEL_CONFIG, TEST_CONFIG
from model import build_model


class Detector:
    """YOLOv8 目标检测器"""

    def __init__(self, weights_path=None, device=None):
        """
        初始化检测器

        Args:
            weights_path: 模型权重路径
            device: 推理设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[INFO] 使用设备: {self.device}")

        self.num_classes = DATASET_CONFIG['num_classes']
        self.class_names = DATASET_CONFIG['classes']
        self.input_size = DATASET_CONFIG['image_size']  # 416
        self.anchors = MODEL_CONFIG['anchors']
        self.strides = [8, 16, 32]  # P3, P4, P5 对应的 stride

        # 为每个尺度生成 anchor grid
        self.anchor_grids = self._make_anchor_grids()

        # 构建模型
        self.model = build_model(num_classes=self.num_classes)

        # 加载权重
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"[INFO] 已加载权重: {weights_path}")
        else:
            print("[WARN] 未加载预训练权重，检测结果可能不准确")

        self.model.to(self.device)
        self.model.eval()

        # 为每个类别生成颜色
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(self.num_classes, 3), dtype=np.uint8).tolist()

    def _make_anchor_grids(self):
        """构建 anchor grids"""
        anchor_grids = []
        for i, stride in enumerate(self.strides):
            grid_h = self.input_size // stride
            grid_w = self.input_size // stride
            anchors = torch.tensor(self.anchors[i], dtype=torch.float32).view(3, 2)
            anchor_grids.append({
                'anchors': anchors,
                'grid_h': grid_h,
                'grid_w': grid_w,
                'stride': stride,
            })
        return anchor_grids

    def preprocess(self, image):
        """
        预处理图片

        Args:
            image: numpy array (H, W, 3) BGR or RGB

        Returns:
            image_tensor: (1, 3, 416, 416)
            original_shape: (H, W) of original image
            scale_info: (scale, pad_x, pad_y)
        """
        if image is None:
            raise ValueError("输入图片为空")

        h, w = image.shape[:2]

        # 计算缩放比例（保持宽高比）
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 缩放图片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到 target_size x target_size
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_y = (self.input_size - new_h) // 2
        pad_x = (self.input_size - new_w) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # 归一化并转换为 tensor
        image_tensor = torch.from_numpy(padded.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 416, 416)

        return image_tensor, (h, w), (scale, pad_x, pad_y)

    def _decode_outputs(self, predictions):
        """
        解码 YOLO 模型输出

        Args:
            predictions: list of 3 tensors [B, 3*25, H, W]

        Returns:
            boxes: (N, 4) [x1, y1, x2, y2] in 416x416 space
            scores: (N,)
            class_ids: (N,)
        """
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for i, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            num_anchors = 3
            num_outputs = self.num_classes + 5  # 25

            # Reshape: [B, 3*25, H, W] -> [B, 3, 25, H, W] -> [B, 3, H, W, 25]
            pred = pred.view(B, num_anchors, num_outputs, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, 3, H, W, 25]

            stride = self.strides[i]
            anchor = self.anchor_grids[i]['anchors'].to(self.device)  # [3, 2]

            # 生成网格坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing='ij'
            )

            # 解码 x, y
            tx = pred[..., 0]  # [B, 3, H, W]
            ty = pred[..., 1]
            tw = pred[..., 2]
            th = pred[..., 3]

            # YOLOv8 风格解码
            bx = (tx.sigmoid() * 2 - 0.5 + grid_x) * stride
            by = (ty.sigmoid() * 2 - 0.5 + grid_y) * stride
            bw = anchor[:, 0].view(1, 3, 1, 1) * (tw.sigmoid() * 2) ** 2
            bh = anchor[:, 1].view(1, 3, 1, 1) * (th.sigmoid() * 2) ** 2

            # 转为中心点 + 宽高 -> x1, y1, x2, y2
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2

            # 置信度和类别分数
            obj_conf = pred[..., 4].sigmoid()  # [B, 3, H, W]
            cls_scores = pred[..., 5:].sigmoid()  # [B, 3, H, W, num_classes]

            # 合并分数: obj * cls_max
            cls_max, cls_id = cls_scores.max(dim=-1)  # [B, 3, H, W]
            score = obj_conf * cls_max  # [B, 3, H, W]

            # 展平
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, 3, H, W, 4]
            boxes = boxes.reshape(B, -1, 4)[0]  # [N, 4]
            score = score.reshape(B, -1)[0]  # [N]
            cls_id = cls_id.reshape(B, -1)[0]  # [N]

            all_boxes.append(boxes)
            all_scores.append(score)
            all_class_ids.append(cls_id)

        # 合并所有尺度
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_class_ids = torch.cat(all_class_ids, dim=0)

        return all_boxes, all_scores, all_class_ids

    def _nms(self, boxes, scores, class_ids, conf_thres=0.25, iou_thres=0.45):
        """
        非极大值抑制 (NMS)

        Args:
            boxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            class_ids: (N,)
            conf_thres: 置信度阈值
            iou_thres: IoU 阈值

        Returns:
            detections: list of [x1, y1, x2, y2, score, class_id]
        """
        # 过滤低置信度
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if boxes.numel() == 0:
            return []

        # 按类别分别做 NMS
        detections = []
        unique_classes = class_ids.unique()

        for cls_id in unique_classes:
            cls_mask = class_ids == cls_id
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            # 按分数排序
            sorted_idx = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[sorted_idx]
            cls_scores = cls_scores[sorted_idx]

            keep = []
            while cls_boxes.shape[0] > 0:
                keep.append(0)
                if cls_boxes.shape[0] == 1:
                    break

                # 计算 IoU
                x1 = torch.max(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = torch.max(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = torch.min(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = torch.min(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                area1 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (cls_boxes[0, 3] - cls_boxes[0, 1])
                area2 = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                iou = inter / (area1 + area2 - inter + 1e-6)

                # 过滤重叠框
                cls_boxes = cls_boxes[1:][iou < iou_thres]
                cls_scores = cls_scores[1:][iou < iou_thres]

            for idx in keep:
                box = cls_boxes[idx].cpu().tolist()
                score = cls_scores[idx].cpu().item()
                detections.append(box + [score, cls_id.cpu().item()])

        return detections

    def detect(self, image, conf_thres=0.25, iou_thres=0.45):
        """
        对图片进行目标检测

        Args:
            image: numpy array (H, W, 3) BGR 图片
            conf_thres: 置信度阈值
            iou_thres: NMS IoU 阈值

        Returns:
            detections: list of [x1, y1, x2, y2, score, class_id]
            image_with_boxes: 绘制了检测框的图片 (numpy array)
        """
        # 预处理
        image_tensor, (orig_h, orig_w), (scale, pad_x, pad_y) = self.preprocess(image)
        image_tensor = image_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # 解码
        boxes, scores, class_ids = self._decode_outputs(predictions)

        # NMS
        detections_416 = self._nms(boxes, scores, class_ids, conf_thres, iou_thres)

        # 将检测结果映射回原始图片坐标
        detections = []
        for det in detections_416:
            x1, y1, x2, y2, score, cls_id = det

            # 从 padded 416x416 映射回原始图片
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # 裁剪到图片边界
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 > x1 and y2 > y1:
                detections.append([x1, y1, x2, y2, score, int(cls_id)])

        # 按置信度降序排列
        detections.sort(key=lambda x: x[4], reverse=True)

        return detections

    def get_class_name(self, class_id):
        """获取类别名称"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"unknown_{class_id}"

    def get_color(self, class_id):
        """获取类别对应颜色"""
        if 0 <= class_id < len(self.colors):
            return tuple(self.colors[class_id])
        return (255, 255, 255)