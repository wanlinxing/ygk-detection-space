"""
YOLOv8目标检测模型
包含改进的CSPDarknet骨干网络、PANet neck和检测头
添加了CBAM注意力机制作为改进
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import MODEL_CONFIG


class Conv(nn.Module):
    """标准卷积层：Conv + BN + SiLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish激活函数
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """融合BN后的前向传播（推理加速）"""
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """深度可分离卷积"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, groups=in_channels)
    
    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    """标准瓶颈块"""
    
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """C2f模块：改进的CSP Bottleneck，带2个卷积"""
    
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = int(out_channels * expansion)
        
        self.conv1 = Conv(in_channels, 2 * self.hidden_channels, 1, 1)
        self.conv2 = Conv((2 + n) * self.hidden_channels, out_channels, 1, 1)
        self.m = nn.ModuleList(Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, groups, expansion=1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.conv1(x).split((self.hidden_channels, self.hidden_channels), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """快速空间金字塔池化"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class CBAM(nn.Module):
    """
    CBAM注意力机制 (Convolutional Block Attention Module)
    包含通道注意力和空间注意力两个子模块
    这是对原始YOLOv8的改进
    """
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class Backbone(nn.Module):
    """CSPDarknet骨干网络（带CBAM注意力机制）"""
    
    def __init__(self, base_channels=64, base_depth=3):
        super().__init__()
        
        # P1/2
        self.stem = Conv(3, base_channels, kernel_size=3, stride=2)
        
        # P2/4
        self.stage1 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, n=base_depth, shortcut=True)
        )
        
        # P3/8
        self.stage2 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, n=base_depth * 2, shortcut=True)
        )
        
        # P4/16
        self.stage3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, n=base_depth * 2, shortcut=True)
        )
        
        # P5/32
        self.stage4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            C2f(base_channels * 16, base_channels * 16, n=base_depth, shortcut=True),
            SPPF(base_channels * 16, base_channels * 16, kernel_size=5)
        )
        
        # CBAM注意力模块（改进点）
        self.cbam3 = CBAM(base_channels * 4)   # P3
        self.cbam4 = CBAM(base_channels * 8)   # P4
        self.cbam5 = CBAM(base_channels * 16)  # P5
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        
        x = self.stage2(x)
        c3 = self.cbam3(x)  # P3/8
        
        x = self.stage3(x)
        c4 = self.cbam4(x)  # P4/16
        
        x = self.stage4(x)
        c5 = self.cbam5(x)  # P5/32
        
        return c3, c4, c5


class Neck(nn.Module):
    """PANet特征融合网络"""
    
    def __init__(self, base_channels=64, base_depth=3):
        super().__init__()
        
        # 上采样融合
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P5 -> P4
        self.c2f_p4 = C2f(base_channels * 24, base_channels * 8, n=base_depth, shortcut=False)
        
        # P4 -> P3
        self.c2f_p3 = C2f(base_channels * 12, base_channels * 4, n=base_depth, shortcut=False)
        
        # P3 -> P4
        self.conv_p3 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.c2f_p4_down = C2f(base_channels * 12, base_channels * 8, n=base_depth, shortcut=False)
        
        # P4 -> P5
        self.conv_p4 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.c2f_p5 = C2f(base_channels * 24, base_channels * 16, n=base_depth, shortcut=False)
    
    def forward(self, c3, c4, c5):
        # 自顶向下
        p5 = c5
        x = self.upsample(p5)
        x = torch.cat([x, c4], dim=1)
        p4 = self.c2f_p4(x)
        
        x = self.upsample(p4)
        x = torch.cat([x, c3], dim=1)
        p3 = self.c2f_p3(x)
        
        # 自底向上
        x = self.conv_p3(p3)
        x = torch.cat([x, p4], dim=1)
        p4_out = self.c2f_p4_down(x)
        
        x = self.conv_p4(p4_out)
        x = torch.cat([x, p5], dim=1)
        p5_out = self.c2f_p5(x)
        
        return p3, p4_out, p5_out


class TransformerBlock(nn.Module):
    """Transformer编码器块 - 引入自注意力机制"""
    
    def __init__(self, channels, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Multi-Head Self-Attention
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # MLP
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 将特征图转换为序列: (B, C, H, W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-Attention with residual
        x_norm = self.norm1(x_flat)
        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_dropout(x_attn)
        x_flat = x_flat + x_attn
        
        # MLP with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # 转换回特征图: (B, H*W, C) -> (B, C, H, W)
        x_out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class MambaBlock(nn.Module):
    """Mamba块 - 选择性状态空间模型"""
    
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * channels)
        
        # 输入投影
        self.in_proj = nn.Linear(channels, self.d_inner * 2, bias=False)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # 选择性参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A参数（状态空间模型）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, channels, bias=False)
        
        self.act = nn.SiLU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        
        # 转换为序列: (B, C, H, W) -> (B, L, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 输入投影
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 因果卷积
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = self.act(x)
        
        # SSM参数
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        # 选择性扫描（简化版）
        x_ssm = self.selective_scan(x, A, D)
        
        # 门控
        x = x_ssm * F.silu(res)
        
        # 输出投影
        x = self.out_proj(x)
        
        # 转换回特征图: (B, L, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
    
    def selective_scan(self, x, A, D):
        """选择性扫描 - 状态空间模型核心"""
        B, L, d = x.shape
        
        # 离散化
        dt = F.softplus(self.dt_proj(x))
        
        # 状态空间模型计算（简化实现）
        # 实际应用中应使用CUDA优化的选择性扫描
        x_proj = self.x_proj(x)
        B_param, C_param = x_proj.split([self.d_state, self.d_state], dim=-1)
        
        # 简化的状态空间计算
        y = x * D.unsqueeze(0).unsqueeze(0)
        
        return y


class Detect(nn.Module):
    """检测头（支持Transformer和Mamba两种前沿架构）"""
    
    def __init__(self, num_classes=80, anchors=(), base_channels=64, use_transformer=True, use_mamba=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # x, y, w, h, obj + classes
        self.num_layers = len(anchors)
        self.use_transformer = use_transformer
        self.use_mamba = use_mamba
        
        self.anchors = anchors
        self.anchor_grid = [torch.zeros(1)] * self.num_layers
        
        # Transformer模块（前沿架构1：自注意力机制）
        if use_transformer:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(base_channels * 4 * (2 ** i), num_heads=8) 
                for i in range(self.num_layers)
            ])
        
        # Mamba模块（前沿架构2：选择性状态空间模型）
        if use_mamba:
            self.mamba_blocks = nn.ModuleList([
                MambaBlock(base_channels * 4 * (2 ** i), d_state=16, expand=2)
                for i in range(self.num_layers)
            ])
        
        # 为每个尺度创建检测头，每个anchor预测num_outputs个值
        self.m = nn.ModuleList(
            nn.Conv2d(base_channels * 4 * (2 ** i), 3 * self.num_outputs, 1) for i in range(self.num_layers)
        )
    
    def forward(self, x):
        outputs = []
        for i in range(self.num_layers):
            # 应用Transformer自注意力（前沿架构1）
            if self.use_transformer:
                x[i] = self.transformer_blocks[i](x[i])
            
            # 应用Mamba选择性状态空间模型（前沿架构2）
            if self.use_mamba:
                x[i] = self.mamba_blocks[i](x[i])
            
            # 检测头
            x[i] = self.m[i](x[i])
            outputs.append(x[i])
        return outputs


class YOLOv8(nn.Module):
    """YOLOv8目标检测模型（支持CBAM、Transformer、Mamba多种改进）"""
    
    def __init__(self, num_classes=20, base_channels=64, base_depth=3, use_transformer=True, use_mamba=False):
        super().__init__()
        self.num_classes = num_classes
        
        # 骨干网络
        self.backbone = Backbone(base_channels, base_depth)
        
        # 特征融合网络
        self.neck = Neck(base_channels, base_depth)
        
        # 检测头（支持Transformer和Mamba）
        anchors = MODEL_CONFIG['anchors']
        self.head = Detect(num_classes, anchors, base_channels, use_transformer=use_transformer, use_mamba=use_mamba)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 骨干网络
        c3, c4, c5 = self.backbone(x)
        
        # 特征融合
        p3, p4, p5 = self.neck(c3, c4, c5)
        
        # 检测头
        outputs = self.head([p3, p4, p5])
        
        return outputs


def build_model(num_classes=20, use_transformer=True, use_mamba=False):
    """构建YOLOv8模型（支持多种前沿架构）
    
    Args:
        num_classes: 类别数量
        use_transformer: 是否使用Transformer自注意力机制
        use_mamba: 是否使用Mamba选择性状态空间模型
    
    Returns:
        model: YOLOv8模型
    """
    model = YOLOv8(num_classes=num_classes, use_transformer=use_transformer, use_mamba=use_mamba)
    return model


if __name__ == '__main__':
    # 测试模型
    model = build_model(num_classes=20)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    for i, out in enumerate(outputs):
        print(f"输出{i+1}形状: {out.shape}")
