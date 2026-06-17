"""
数据集处理模块
支持Pascal VOC 2012数据集的加载、预处理和增强
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import math
from config import DATASET_CONFIG, AUGMENT_CONFIG


class VOCDataset(Dataset):
    
    def __init__(self, data_dir, split='train', transform=None, augment=False):

        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.augment = augment
        self.image_size = DATASET_CONFIG['image_size']
        self.classes = DATASET_CONFIG['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_ids = self._load_image_ids()
        
    def _load_image_ids(self):
        split_file = os.path.join(self.data_dir, 'ImageSets', 'Main', f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            image_dir = os.path.join(self.data_dir, 'JPEGImages')
            return [os.path.splitext(f)[0] for f in os.listdir(image_dir) 
                    if f.endswith('.jpg')]
    
    def _parse_xml(self, xml_path):
        """解析XML标注文件"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # 获取类别名称
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
        
        return np.array(boxes), np.array(labels)
    
    def _load_image(self, image_path):
        """加载图像"""
        # 使用numpy读取，支持中文路径
        with open(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _augment(self, image, boxes):
        """数据增强"""
        if not self.augment:
            return image, boxes
        
        # HSV色彩增强
        if random.random() < 0.5:
            image = self._hsv_augment(image)
        
        # 水平翻转
        if random.random() < AUGMENT_CONFIG['fliplr']:
            image, boxes = self._flip_lr(image, boxes)
        
        # 随机缩放和平移
        if random.random() < 0.5:
            image, boxes = self._random_perspective(image, boxes)
        
        return image, boxes
    
    def _hsv_augment(self, image):
        """HSV色彩增强"""
        hgain = AUGMENT_CONFIG['hsv_h']
        sgain = AUGMENT_CONFIG['hsv_s']
        vgain = AUGMENT_CONFIG['hsv_v']
        
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
        
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
        
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
    
    def _flip_lr(self, image, boxes):
        """水平翻转"""
        image = np.fliplr(image).copy()
        h, w = image.shape[:2]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes
    
    def _random_perspective(self, image, boxes, degrees=0, translate=0.1, scale=0.5, shear=0):
        """随机透视变换"""
        h, w = image.shape[:2]
        
        # 随机变换矩阵
        M = self._get_transform_matrix(w, h, degrees, translate, scale, shear)
        
        # 应用变换
        image = cv2.warpAffine(image, M[:2], (w, h), borderValue=(114, 114, 114))
        
        # 变换边界框
        if len(boxes) > 0:
            n = len(boxes)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = xy[:, :2].reshape(n, 8)
            
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        
        return image, boxes
    
    def _get_transform_matrix(self, w, h, degrees, translate, scale, shear):
        """获取变换矩阵"""
        # 平移
        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2
        
        # 缩放和旋转
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        
        # 剪切
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        
        # 平移回去
        T = np.eye(3)
        T[0, 2] = w / 2 + random.uniform(-translate, translate) * w
        T[1, 2] = h / 2 + random.uniform(-translate, translate) * h
        
        M = T @ S @ R @ C
        return M
    
    def _resize_and_pad(self, image, boxes):
        """调整图像大小并填充"""
        h, w = image.shape[:2]
        target_size = self.image_size
        
        # 计算缩放比例
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 调整图像大小
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充图像
        padded_image = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        
        # 计算填充位置
        dh, dw = (target_size - new_h) // 2, (target_size - new_w) // 2
        padded_image[dh:dh+new_h, dw:dw+new_w] = image
        
        # 调整边界框
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
        
        return padded_image, boxes, scale, (dw, dh)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        image_id = self.image_ids[idx]
        
        # 加载图像
        image_path = os.path.join(self.data_dir, 'JPEGImages', f'{image_id}.jpg')
        image = self._load_image(image_path)
        
        # 加载标注
        xml_path = os.path.join(self.data_dir, 'Annotations', f'{image_id}.xml')
        boxes, labels = self._parse_xml(xml_path)
        
        # 数据增强
        image, boxes = self._augment(image, boxes)
        
        # 调整大小和填充
        image, boxes, scale, pad = self._resize_and_pad(image, boxes)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # 构建目标张量 [num_boxes, 6] -> [class, x_center, y_center, width, height, image_idx]
        targets = []
        if len(boxes) > 0:
            # 转换为YOLO格式 (x_center, y_center, width, height)
            h, w = image.shape[1:]
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            
            for box, label in zip(boxes, labels):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                targets.append([label, x_center, y_center, width, height])
        
        targets = torch.FloatTensor(targets) if targets else torch.zeros((0, 5))
        
        return image, targets


def collate_fn(batch):
    """自定义批处理函数"""
    images, targets = zip(*batch)

    # 堆叠图像
    images = torch.stack(images, 0)

    # 为每个 target 添加 batch index（修复：之前覆盖了 class_id）
    new_targets = []
    for i, target in enumerate(targets):
        if len(target) > 0:
            # target 格式: [class, x_center, y_center, width, height]
            # 添加 batch index 作为第一列 → [batch_idx, class, x, y, w, h]
            batch_idx = torch.full((len(target), 1), i, dtype=target.dtype)
            target_with_idx = torch.cat([batch_idx, target], dim=1)
            new_targets.append(target_with_idx)

    if new_targets:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.zeros((0, 6))

    return images, targets


def create_dataloader(data_dir, split='train', batch_size=8, augment=False):
    """创建数据加载器"""
    dataset = VOCDataset(data_dir, split=split, augment=augment)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,  # Windows下设为0避免多进程问题
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    import math
    data_dir = DATASET_CONFIG['data_dir']
    
    if os.path.exists(data_dir):
        dataset = VOCDataset(data_dir, split='train', augment=True)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试加载一个样本
        image, targets = dataset[0]
        print(f"图像形状: {image.shape}")
        print(f"目标数量: {len(targets)}")
        print(f"目标示例: {targets[0] if len(targets) > 0 else 'None'}")
    else:
        print(f"数据集目录不存在: {data_dir}")
        print("请先下载VOC2012数据集")
