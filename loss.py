"""
YOLOv8损失函数
包含分类损失、边界框回归损失（CIoU）和目标置信度损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BBoxLoss(nn.Module):
    """边界框回归损失（使用CIoU）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_boxes, target_boxes, anchor_boxes):
        """
        计算CIoU损失
        Args:
            pred_boxes: 预测的边界框 [N, 4] (x, y, w, h)
            target_boxes: 目标边界框 [N, 4] (x, y, w, h)
            anchor_boxes: 锚框 [N, 2] (w, h)
        Returns:
            loss: CIoU损失
        """
        # 将预测值转换为边界框坐标
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = pred_area + target_area - inter_area + 1e-7
        
        # IoU
        iou = inter_area / union_area
        
        # 计算中心点距离
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # 计算最小外接矩形
        c_x1 = torch.min(pred_x1, target_x1)
        c_y1 = torch.min(pred_y1, target_y1)
        c_x2 = torch.max(pred_x2, target_x2)
        c_y2 = torch.max(pred_y2, target_y2)
        
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-7
        
        # 长宽比一致性
        pred_w = pred_boxes[:, 2]
        pred_h = pred_boxes[:, 3]
        target_w = target_boxes[:, 2]
        target_h = target_boxes[:, 3]
        
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7)), 2)
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - (center_distance / c_area + alpha * v)
        
        loss = 1 - ciou
        
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss用于解决类别不平衡问题"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测概率 [N, C]
            target: 目标类别 [N]
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class YOLOLoss(nn.Module):
    """YOLOv8总损失函数"""
    
    def __init__(self, num_classes=20, box_weight=7.5, cls_weight=0.5, dfl_weight=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BBoxLoss()
        self.focal_loss = FocalLoss()
        
        # 用于DFL（Distribution Focal Loss）的卷积核
        self.project = torch.arange(16, dtype=torch.float)
    
    def forward(self, predictions, targets):
        """
        计算总损失
        Args:
            predictions: 模型输出列表 [P3, P4, P5]
            targets: 目标标注 [N, 6] -> [img_idx, class, x, y, w, h]
        """
        device = predictions[0].device
        
        # 初始化损失
        l_cls = torch.tensor(0.0, device=device)
        l_box = torch.tensor(0.0, device=device)
        l_obj = torch.tensor(0.0, device=device)
        
        # 获取batch大小
        batch_size = predictions[0].shape[0]
        
        # 构建目标
        for i, pred in enumerate(predictions):
            # pred shape: [batch, num_anchors, num_outputs, h, w]
            bs, _, ny, nx = pred.shape
            
            # 分离预测值
            pred = pred.view(bs, 3, self.num_classes + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 创建网格
            grid_x = torch.arange(nx, device=device).view(1, 1, 1, nx).float()
            grid_y = torch.arange(ny, device=device).view(1, 1, ny, 1).float()
            
            # 计算目标掩码
            if targets.shape[0] == 0:
                # 如果没有目标，只计算背景损失
                obj_target = torch.zeros_like(pred[..., 4])
                l_obj += self.bce(pred[..., 4], obj_target).mean()
                continue
            
            # 匹配目标到当前特征层
            matched_targets = self._match_targets(targets, i, nx, ny, device)
            
            if matched_targets is None:
                obj_target = torch.zeros_like(pred[..., 4])
                l_obj += self.bce(pred[..., 4], obj_target).mean()
                continue
            
            # 提取匹配的预测和目标
            pred_boxes = matched_targets['pred_boxes']
            target_boxes = matched_targets['target_boxes']
            target_cls = matched_targets['target_cls']
            obj_mask = matched_targets['obj_mask']
            
            # 边界框回归损失
            if len(pred_boxes) > 0:
                l_box += self.bbox_loss(pred_boxes, target_boxes, None)
                
                # 分类损失
                pred_cls = matched_targets['pred_cls']
                if len(pred_cls) > 0:
                    l_cls += self.focal_loss(pred_cls, target_cls)
            
            # 目标置信度损失
            obj_target = torch.zeros_like(pred[..., 4])
            obj_target[obj_mask] = 1.0
            l_obj += self.bce(pred[..., 4], obj_target).mean()
        
        # 加权求和
        loss = self.box_weight * l_box + self.cls_weight * l_cls + l_obj
        
        return {
            'loss': loss,
            'box_loss': l_box,
            'cls_loss': l_cls,
            'obj_loss': l_obj
        }
    
    def _match_targets(self, targets, layer_idx, nx, ny, device):
        """
        将目标匹配到特征层
        简化的实现，实际应使用更复杂的分配策略
        """
        if targets.shape[0] == 0:
            return None
        
        # 根据特征层大小筛选目标
        # 小目标 -> P3, 中目标 -> P4, 大目标 -> P5
        scale_ranges = [
            (0, 32),    # P3
            (32, 64),   # P4
            (64, 128)   # P5
        ]
        
        min_scale, max_scale = scale_ranges[layer_idx]
        
        matched = []
        for t in targets:
            if len(t) == 6:
                img_idx, cls, x, y, w, h = t
            else:
                # 如果目标只有5个值 [class, x, y, w, h]，跳过
                continue
            
            # 计算目标大小（相对于特征图）
            target_size = max(w, h) * nx
            
            if min_scale <= target_size < max_scale or (layer_idx == 2 and target_size >= max_scale):
                # 计算网格位置
                grid_x = int(x * nx)
                grid_y = int(y * ny)
                
                if 0 <= grid_x < nx and 0 <= grid_y < ny:
                    matched.append({
                        'img_idx': int(img_idx),
                        'cls': int(cls),
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'grid_x': grid_x,
                        'grid_y': grid_y
                    })
        
        if len(matched) == 0:
            return None
        
        # 构建返回字典
        result = {
            'pred_boxes': torch.zeros((len(matched), 4), device=device),
            'target_boxes': torch.zeros((len(matched), 4), device=device),
            'target_cls': torch.zeros(len(matched), dtype=torch.long, device=device),
            'pred_cls': torch.zeros((len(matched), self.num_classes), device=device),
            'obj_mask': torch.zeros((len(targets), 3, ny, nx), dtype=torch.bool, device=device)
        }
        
        for i, m in enumerate(matched):
            result['target_boxes'][i] = torch.tensor([m['x'], m['y'], m['w'], m['h']], device=device)
            result['target_cls'][i] = m['cls']
            result['obj_mask'][m['img_idx'], :, m['grid_y'], m['grid_x']] = True
        
        return result


class ComputeLoss:
    """简化版的损失计算"""
    
    def __init__(self, model, num_classes=20):
        self.model = model
        self.num_classes = num_classes
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.bbox_loss = BBoxLoss()
    
    def __call__(self, predictions, targets):
        device = predictions[0].device
        l_cls = torch.zeros(1, device=device)
        l_box = torch.zeros(1, device=device)
        l_obj = torch.zeros(1, device=device)
        
        # 简化处理：直接计算各层损失
        for pred in predictions:
            bs, _, h, w = pred.shape
            pred = pred.view(bs, 3, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
            # 这里简化处理，实际应该做正负样本分配
            # 仅演示损失计算流程
            if targets.shape[0] > 0:
                # 模拟一些正样本
                obj_target = torch.zeros_like(pred[..., 4])
                l_obj += self.bce_obj(pred[..., 4], obj_target).mean()
        
        loss = l_box + l_cls + l_obj
        
        return loss * bs, torch.cat((l_box, l_cls, l_obj)).detach()


if __name__ == '__main__':
    # 测试损失函数
    from model import build_model
    
    model = build_model(num_classes=20)
    criterion = YOLOLoss(num_classes=20)
    
    # 模拟输入
    x = torch.randn(2, 3, 640, 640)
    predictions = model(x)
    
    # 模拟目标
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.3],
        [0, 5, 0.3, 0.7, 0.1, 0.2],
        [1, 10, 0.6, 0.4, 0.3, 0.4]
    ])
    
    loss_dict = criterion(predictions, targets)
    
    print(f"总损失: {loss_dict['loss']:.4f}")
    print(f"边界框损失: {loss_dict['box_loss']:.4f}")
    print(f"分类损失: {loss_dict['cls_loss']:.4f}")
    print(f"置信度损失: {loss_dict['obj_loss']:.4f}")
