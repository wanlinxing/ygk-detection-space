"""
训练脚本
实现YOLOv8模型的训练流程，包括数据加载、模型训练、验证和保存
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, VAL_CONFIG
from model import build_model
from dataset import create_dataloader
from loss import YOLOLoss


class Trainer:
    """训练器类"""

    def __init__(self):
        self.device = torch.device(TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建模型
        self.model = build_model(num_classes=DATASET_CONFIG['num_classes'])
        self.model.to(self.device)

        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型总参数量: {total_params / 1e6:.2f}M")

        # 损失函数
        self.criterion = YOLOLoss(num_classes=DATASET_CONFIG['num_classes'])

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAIN_CONFIG['lr'],
            weight_decay=TRAIN_CONFIG['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=TRAIN_CONFIG['epochs'],
            eta_min=TRAIN_CONFIG['lr'] * 0.01
        )

        # 训练记录
        self.epoch = 0
        self.best_map = 0.0
        self.train_losses = []
        self.val_maps = []

        # 创建保存目录
        os.makedirs(TRAIN_CONFIG['save_dir'], exist_ok=True)

        # TensorBoard（使用英文路径避免中文路径问题）
        try:
            log_dir = os.path.join(TRAIN_CONFIG['save_dir'], 'tb_logs')
            if os.path.exists(log_dir) and os.path.isfile(log_dir):
                os.remove(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except Exception as e:
            print(f"TensorBoard初始化失败: {e}")
            print("训练将继续，但不记录TensorBoard日志")
            self.writer = None

    def train_one_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_box_loss = 0
        total_cls_loss = 0
        total_obj_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}/{TRAIN_CONFIG['epochs']}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            predictions = self.model(images)

            # 计算损失
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            total_box_loss += loss_dict['box_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_dict["box_loss"].item():.4f}',
                'cls': f'{loss_dict["cls_loss"].item():.4f}',
                'obj': f'{loss_dict["obj_loss"].item():.4f}'
            })

        avg_loss = total_loss / len(dataloader)
        avg_box_loss = total_box_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_obj_loss = total_obj_loss / len(dataloader)

        return {
            'loss': avg_loss,
            'box_loss': avg_box_loss,
            'cls_loss': avg_cls_loss,
            'obj_loss': avg_obj_loss
        }

    @torch.no_grad()
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_targets = []

        pbar = tqdm(dataloader, desc="Validation")

        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            predictions = self.model(images)

            # 计算损失
            loss_dict = self.criterion(predictions, targets)
            total_loss += loss_dict['loss'].item()

            # 收集预测和目标用于计算mAP
            # 这里简化处理，实际应解码预测结果

        avg_loss = total_loss / len(dataloader)

        # 计算mAP（简化版）
        map50 = 0.0  # 实际应计算

        return {'loss': avg_loss, 'mAP50': map50}

    def save_checkpoint(self, filename='last.pt', is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'train_losses': self.train_losses,
            'val_maps': self.val_maps
        }

        path = os.path.join(TRAIN_CONFIG['save_dir'], filename)
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(TRAIN_CONFIG['save_dir'], 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型，mAP: {self.best_map:.4f}")

    def load_checkpoint(self, filename='last.pt'):
        """加载检查点"""
        path = os.path.join(TRAIN_CONFIG['save_dir'], filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_map = checkpoint['best_map']
            self.train_losses = checkpoint['train_losses']
            self.val_maps = checkpoint['val_maps']
            print(f"加载检查点: {path}，从epoch {self.epoch}继续训练")
            return True
        return False

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        epochs = range(1, len(self.train_losses) + 1)

        # 总损失
        axes[0, 0].plot(epochs, [loss['loss'] for loss in self.train_losses], 'b-', label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 边界框损失
        axes[0, 1].plot(epochs, [loss['box_loss'] for loss in self.train_losses], 'r-', label='Box Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Bounding Box Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 分类损失
        axes[1, 0].plot(epochs, [loss['cls_loss'] for loss in self.train_losses], 'g-', label='Cls Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # mAP
        if self.val_maps:
            axes[1, 1].plot(range(1, len(self.val_maps) + 1), self.val_maps, 'm-', label='mAP@0.5')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP')
            axes[1, 1].set_title('Validation mAP')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(TRAIN_CONFIG['save_dir'], 'training_curves.png'), dpi=300)
        plt.close()

    def train(self):
        """主训练循环"""
        # 加载数据
        data_dir = DATASET_CONFIG['data_dir']

        if not os.path.exists(data_dir):
            print(f"错误：数据集目录不存在: {data_dir}")
            print("请先下载VOC2012数据集")
            return

        train_loader = create_dataloader(
            data_dir,
            split='train',
            batch_size=TRAIN_CONFIG['batch_size'],
            augment=True
        )

        val_loader = create_dataloader(
            data_dir,
            split='val',
            batch_size=VAL_CONFIG['batch_size'],
            augment=False
        )

        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")

        # 尝试加载已有检查点
        self.load_checkpoint('last.pt')

        # 训练循环
        for epoch in range(self.epoch, TRAIN_CONFIG['epochs']):
            self.epoch = epoch + 1

            print(f"\n{'=' * 50}")
            print(f"Epoch {self.epoch}/{TRAIN_CONFIG['epochs']}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'=' * 50}")

            # 训练
            train_metrics = self.train_one_epoch(train_loader)
            self.train_losses.append(train_metrics)

            print(f"训练损失: {train_metrics['loss']:.4f}")
            print(f"  - 边界框损失: {train_metrics['box_loss']:.4f}")
            print(f"  - 分类损失: {train_metrics['cls_loss']:.4f}")
            print(f"  - 置信度损失: {train_metrics['obj_loss']:.4f}")

            # 记录到TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], self.epoch)
                self.writer.add_scalar('Loss/box', train_metrics['box_loss'], self.epoch)
                self.writer.add_scalar('Loss/cls', train_metrics['cls_loss'], self.epoch)
                self.writer.add_scalar('Loss/obj', train_metrics['obj_loss'], self.epoch)
            
            # 验证
            if self.epoch % 5 == 0 or self.epoch == TRAIN_CONFIG['epochs']:
                val_metrics = self.validate(val_loader)
                self.val_maps.append(val_metrics['mAP50'])
                
                print(f"验证损失: {val_metrics['loss']:.4f}")
                print(f"验证mAP@0.5: {val_metrics['mAP50']:.4f}")
                
                if self.writer is not None:
                    self.writer.add_scalar('Loss/val', val_metrics['loss'], self.epoch)
                    self.writer.add_scalar('mAP/val', val_metrics['mAP50'], self.epoch)

                # 保存最佳模型
                if val_metrics['mAP50'] > self.best_map:
                    self.best_map = val_metrics['mAP50']
                    self.save_checkpoint('best.pt', is_best=True)

            # 更新学习率
            self.scheduler.step()

            # 保存检查点
            if self.epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{self.epoch}.pt')

            self.save_checkpoint('last.pt')

            # 绘制训练曲线
            self.plot_training_curves()

        print("\n训练完成!")
        print(f"最佳mAP@0.5: {self.best_map:.4f}")

        if self.writer is not None:
            self.writer.close()


def main():
    """主函数"""
    print("=" * 50)
    print("YOLOv8目标检测训练")
    print("=" * 50)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建训练器
    trainer = Trainer()

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()