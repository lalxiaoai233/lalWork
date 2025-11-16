"""
CBDES MoE: 分层解耦专家混合模型用于自动驾驶功能模块

这个模块实现了CBDES MoE架构，包含异构专家网络和轻量级自注意力路由器（SAR）用于动态专家选择。

主要特性：
1. 四种异构专家网络：Swin Transformer、ResNet、ConvNeXt、PVT
2. 自注意力路由器（SAR）进行动态专家选择
3. 负载均衡正则化确保专家均匀使用
4. 支持ImageNet预训练权重加载
5. 多尺度特征输出

作者：liuailin
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, load_checkpoint
from mmdet.models import BACKBONES
from mmdet.models.backbones import ResNet, SwinTransformer
from mmcv.cnn import build_norm_layer, build_activation_layer


def load_pretrained_weights(model, pretrained_path, strict=False):
    """
    为专家网络加载预训练权重
    
    支持多种检查点格式，自动过滤分类头权重，只保留骨干网络权重。
    这是确保CBDES MoE能正确加载预训练专家网络的关键函数。
    
    Args:
        model: 要加载权重的模型
        pretrained_path: 预训练权重文件路径
        strict: 是否严格匹配所有键名
        
    Returns:
        model: 加载权重后的模型
    """
    # 检查预训练权重路径是否存在
    if pretrained_path is None or not os.path.exists(pretrained_path):
        print(f"警告: 预训练权重未找到 {pretrained_path}")
        return model
    
    try:
        # 加载检查点文件到CPU内存
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 处理不同的检查点格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉分类头权重，只保留骨干网络权重
        filtered_state_dict = {}
        for key, value in state_dict.items():
            # 跳过分类头层（包含classifier、head、fc、linear等关键词）
            if any(skip_key in key.lower() for skip_key in ['classifier', 'head', 'fc', 'linear']):
                continue
            filtered_state_dict[key] = value
        
        # 加载过滤后的权重到模型
        model.load_state_dict(filtered_state_dict, strict=strict)
        print(f"成功加载预训练权重: {pretrained_path}")
        
    except Exception as e:
        print(f"加载预训练权重时出错: {e}")
        print("继续使用随机初始化...")
    
    return model


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt块实现，用于专家网络
    
    ConvNeXt是现代卷积架构，结合了深度可分离卷积、层归一化和层缩放技术。
    这个块是ConvNeXt专家网络的基本构建单元。
    """
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        """
        初始化ConvNeXt块
        
        Args:
            dim: 特征维度
            drop_path: DropPath正则化概率
            layer_scale_init_value: 层缩放初始化值
        """
        super().__init__()
        # 深度可分离卷积（7x7卷积核）
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 层归一化
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # 第一个点卷积（1x1卷积的线性层形式）
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 512
        # GELU激活函数
        self.act = nn.GELU()
        # 第二个点卷积
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 层缩放参数（可选的残差缩放）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        # DropPath正则化
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征张量 (B, C, H, W)
            
        Returns:
            x: 输出特征张量 (B, C, H, W)
        """
        input = x
        # 深度可分离卷积
        x = self.dwconv(x)
        # 转换维度用于层归一化 (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # 第一个点卷积和激活
        x = self.pwconv1(x)
        x = self.act(x)
        # 第二个点卷积
        x = self.pwconv2(x)
        # 应用层缩放
        if self.gamma is not None:
            x = self.gamma * x

        # 转换回原始维度 (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # 残差连接
        x = input + self.drop_path(x)
        return x


class ConvNeXtExpert(nn.Module):
    """
    ConvNeXt专家网络，支持预训练权重
    
    这是CBDES MoE中的ConvNeXt专家网络实现，支持从torchvision、timm或自定义检查点加载预训练权重。
    网络采用分层结构，包含stem层、多个下采样层和ConvNeXt块。
    """
    
    def __init__(self, in_channels=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[1, 2, 3],
                 pretrained=None):
        """
        初始化ConvNeXt专家网络
        
        Args:
            in_channels: 输入通道数
            depths: 每个阶段的块数量
            dims: 每个阶段的特征维度
            drop_path_rate: DropPath正则化率
            layer_scale_init_value: 层缩放初始化值
            out_indices: 输出特征图的索引
            pretrained: 预训练权重路径或'torchvision'/'timm'
        """
        super().__init__()
        self.out_indices = out_indices
        self.pretrained = pretrained
        
        # 构建stem层（初始下采样层）
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),  # 4x4卷积，步长4
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # 构建下采样层（阶段间的过渡层）
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),  # 2x2卷积，步长2
            )
            self.downsample_layers.append(downsample_layer)
        
        # 构建ConvNeXt块（每个阶段的核心处理单元）
        self.stages = nn.ModuleList()
        # 计算DropPath率（线性递增）
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # 输出归一化层
        self.norm = nn.BatchNorm2d(dims[-1])
        
        # 如果指定了预训练权重，则加载
        if self.pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained ConvNeXt weights."""
        if self.pretrained == 'torchvision':
            # Use torchvision's pretrained ConvNeXt
            try:
                # Try different ConvNeXt model names
                if hasattr(models, 'convnext_tiny'):
                    pretrained_model = models.convnext_tiny(pretrained=True)
                elif hasattr(models, 'convnext_small'):
                    pretrained_model = models.convnext_small(pretrained=True)
                else:
                    print("ConvNeXt not available in this torchvision version, skipping pretrained weights")
                    print("💡 Solution: Upgrade torchvision to 0.13.0+ or use custom pretrained weights")
                    return
                
                # Extract backbone weights (exclude classifier)
                pretrained_state_dict = {}
                for name, param in pretrained_model.named_parameters():
                    if 'classifier' not in name:
                        pretrained_state_dict[name] = param.data
                
                # Load compatible weights
                model_state_dict = self.state_dict()
                for name, param in pretrained_state_dict.items():
                    if name in model_state_dict and param.shape == model_state_dict[name].shape:
                        model_state_dict[name] = param
                
                self.load_state_dict(model_state_dict, strict=False)
                print("Successfully loaded torchvision ConvNeXt pretrained weights")
            except Exception as e:
                print(f"Failed to load torchvision ConvNeXt weights: {e}")
        elif self.pretrained == 'timm':
            # Use timm library for ConvNeXt pretrained weights
            try:
                import timm
                pretrained_model = timm.create_model('convnext_tiny', pretrained=True)
                
                # Extract backbone weights (exclude classifier)
                pretrained_state_dict = {}
                for name, param in pretrained_model.named_parameters():
                    if 'head' not in name and 'classifier' not in name:
                        pretrained_state_dict[name] = param.data
                
                # Load compatible weights
                model_state_dict = self.state_dict()
                for name, param in pretrained_state_dict.items():
                    if name in model_state_dict and param.shape == model_state_dict[name].shape:
                        model_state_dict[name] = param
                
                self.load_state_dict(model_state_dict, strict=False)
                print("Successfully loaded timm ConvNeXt pretrained weights")
            except ImportError:
                print("timm library not available. Install with: pip install timm")
            except Exception as e:
                print(f"Failed to load timm ConvNeXt weights: {e}")
        else:
            # Load from custom checkpoint
            load_pretrained_weights(self, self.pretrained)
        
    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outputs.append(x)
        
        return outputs


class PyramidVisionTransformerBlock(nn.Module):
    """Pyramid Vision Transformer Block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PyramidVisionTransformerExpert(nn.Module):
    """Pyramid Vision Transformer Expert Network with pretrained support."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], out_indices=[1, 2, 3],
                 pretrained=None):
        super().__init__()
        self.out_indices = out_indices
        self.pretrained = pretrained
        
        # Patch embeddings
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        
        for i in range(len(embed_dims)):
            if i == 0:
                patch_embed = nn.Conv2d(in_channels, embed_dims[i], kernel_size=patch_size, stride=patch_size)
            else:
                patch_embed = nn.Conv2d(embed_dims[i-1], embed_dims[i], kernel_size=2, stride=2)
            self.patch_embeds.append(patch_embed)
            
            pos_embed = nn.Parameter(torch.zeros(1, (img_size // (patch_size * (2 ** i))) ** 2, embed_dims[i]))
            self.register_parameter(f'pos_embed_{i}', pos_embed)
            self.pos_embeds.append(pos_embed)
            self.pos_drops.append(nn.Dropout(p=drop_rate))
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(len(embed_dims)):
            block = nn.ModuleList([
                PyramidVisionTransformerBlock(
                    dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j], norm_layer=norm_layer
                ) for j in range(depths[i])
            ])
            self.blocks.append(block)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        
        # Load pretrained weights if specified
        if self.pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pretrained PVT weights."""
        if self.pretrained:
            load_pretrained_weights(self, self.pretrained)
        
    def forward(self, x):
        B = x.shape[0]
        outputs = []
        
        for i in range(len(self.patch_embeds)):
            x = self.patch_embeds[i](x)
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # B, N, C
            # Dynamic position encoding based on actual spatial dimensions
            pos_embed = self.pos_embeds[i]
            N = x.shape[1]
            if pos_embed.shape[1] != N:
                # interpolate absolute pos_embed to match current HxW
                N0, C = pos_embed.shape[1], pos_embed.shape[2]
                H0 = int(round(N0 ** 0.5))
                W0 = max(1, N0 // H0)
                pe = pos_embed[0].transpose(0, 1).reshape(C, H0, W0).unsqueeze(0)
                pe = F.interpolate(pe, size=(H, W), mode='bicubic', align_corners=False)
                pos_embed = pe[0].reshape(C, H * W).transpose(0, 1).unsqueeze(0)
            else:
                pos_embed = pos_embed[:, :N, :]
            x = x + pos_embed
            x = self.pos_drops[i](x)
            
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            
            if i in self.out_indices:
                x_norm = x  # Skip normalization for now to avoid dimension issues
                x_norm = x_norm.transpose(1, 2).reshape(B, -1, H, W)
                outputs.append(x_norm)
            
            # Reshape back to 4D for next stage
            x = x.transpose(1, 2).reshape(B, -1, H, W)
        
        return outputs


class SelfAttentionRouter(nn.Module):
    """
    轻量级自注意力路由器（SAR）用于专家选择
    
    这是CBDES MoE的核心组件，负责根据输入特征动态选择最合适的专家网络。
    采用三步卷积池化 + 自注意力编码 + 专家评分的完整流程。
    
    主要功能：
    1. 三步卷积和池化处理：逐步提取和压缩图像特征
    2. 自注意力编码：将特征转换为token序列并进行多头自注意力处理
    3. 图像级嵌入：通过token平均得到全局语义信息
    4. 专家评分：3层MLP输出专家logits，通过softmax转换为路由概率
    5. 负载均衡正则化：确保专家均匀使用
    
    处理流程：
    输入图像 -> 三步卷积池化 -> token序列 -> 多头自注意力 -> 图像级嵌入 -> 专家logits -> 路由概率
    """
    
    def __init__(self, input_dim, num_experts=4, embedding_dim=128, num_heads=8, dropout=0.1):
        """
        初始化自注意力路由器
        
        Args:
            input_dim: 输入特征维度
            num_experts: 专家网络数量
            embedding_dim: 嵌入维度
            num_heads: 注意力头数（当前未使用，保留接口）
            dropout: Dropout概率
        """
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        
        # 三步卷积和池化处理模块
        self.conv_modules = nn.ModuleList()
        
        # 第一步：3x3卷积 + BN + PReLU + 2x2最大池化（步长2）
        self.conv_modules.append(nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ))
        
        # 第二步：3x3卷积 + BN + PReLU + 2x2最大池化（步长2）
        self.conv_modules.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ))
        
        # 第三步：3x3卷积 + BN + PReLU + 2x2最大池化（步长2）
        self.conv_modules.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ))
        
        # 自注意力编码模块
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=128,  # d_emb = 128
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(128)
        
        # 第三步：专家评分MLP（3层带PReLU激活）
        self.expert_scorer = nn.Sequential(
            nn.Linear(128, 64),  # 第一层：128 -> 64（第一次压缩）
            nn.PReLU(),
            nn.Linear(64, 32),  # 第二层：64 -> 32（第二次压缩）
            nn.PReLU(),
            nn.Linear(32, num_experts),  # 第三层：32 -> 4（专家logits）
        )
        
        # 负载均衡正则化：跟踪专家使用情况
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0))
        
    def forward(self, x):
        """
        前向传播：通过三步卷积池化 + 自注意力编码 + 专家评分计算路由概率和损失
        
        处理流程：
        1. 输入图像特征图 X (B, C, H, W)
        2. 三步卷积和池化处理：每步包含3x3卷积、BN、PReLU激活和2x2最大池化，得到特征X3
        3. 自注意力编码：
           - 将X3展平成token序列T (B, N, d_emb)，其中N=H*W，d_emb=128
           - 对T进行多头自注意力MHA和层归一化，得到T'
           - 对T'的token维度N取平均，得到图像级嵌入G (B, d_emb)
        4. 专家评分：
           - 将G输入3层带PReLU激活的MLP，输出专家logits S (B, num_experts)
           - 通过softmax将S转换为路由概率P (B, num_experts)
        5. 负载均衡损失计算
        
        Args:
            x: 输入特征张量 (B, C, H, W)
            
        Returns:
            P: 路由概率 (B, num_experts)，表示每张图像分配给各专家的概率
            routing_loss: 负载均衡损失
        """
        B, C, H, W = x.shape
        
        # 第一步：三步卷积和池化处理，得到特征X3
        x3 = x
        for conv_module in self.conv_modules:
            x3 = conv_module(x3)  # (B, 128, H', W')
        
        # 第二步：自注意力编码
        B, C, H, W = x3.shape
        
        # 将特征X3展平成token序列T
        T = x3.flatten(2).transpose(1, 2)  # (B, N, d_emb) 其中 N = H*W, d_emb = 128
        
        # 多头自注意力处理
        T_attended, _ = self.multihead_attn(T, T, T)  # (B, N, d_emb)
        
        # 残差连接和层归一化
        T_prime = self.layer_norm(T + T_attended)  # (B, N, d_emb)
        
        # 对token维度N取平均，得到图像级嵌入G
        G = T_prime.mean(dim=1)  # (B, d_emb) = (B, 128)
        
        # 第三步：专家评分，基于图像级嵌入G计算专家logits和路由概率
        S = self.expert_scorer(G)  # 专家logits (B, num_experts)
        P = F.softmax(S, dim=-1)  # 路由概率 (B, num_experts)
        
        # 计算负载均衡损失
        routing_loss = self._compute_load_balance_loss(P)
        
        return P, routing_loss
    
    def _compute_load_balance_loss(self, expert_weights):
        """
        计算负载均衡正则化损失
        
        通过监控专家使用情况，确保所有专家被均匀使用，
        避免某些专家被过度使用而其他专家被忽略。
        
        Args:
            expert_weights: 专家权重 (B, num_experts)
            
        Returns:
            load_balance_loss: 负载均衡损失
        """
        if not self.training:
            return torch.tensor(0.0, device=expert_weights.device)
        
        # 更新专家使用计数
        expert_selections = torch.argmax(expert_weights, dim=-1)  # (B,) 选择权重最大的专家
        for i in range(self.num_experts):
            self.expert_counts[i] += (expert_selections == i).sum().float()
        self.total_samples += expert_weights.shape[0]
        
        # 计算负载均衡损失：专家使用概率的方差
        expert_probs = self.expert_counts / (self.total_samples + 1e-8)
        load_balance_loss = torch.var(expert_probs) * self.num_experts
        
        return load_balance_loss


@BACKBONES.register_module()
class CBDESMoE(nn.Module):
    """
    CBDES MoE: 分层解耦专家混合模型用于BEV感知
    
    这是CBDES MoE的主要实现，集成了多个结构异构的专家网络
    和轻量级自注意力路由器，用于动态专家选择。
    
    架构特点：
    1. 四种异构专家网络：Swin Transformer、ResNet、ConvNeXt、PVT
    2. 自注意力路由器（SAR）进行智能专家选择
    3. 负载均衡正则化确保专家均匀使用
    4. 支持ImageNet预训练权重加载
    5. 多尺度特征输出适配下游任务
    6. 动态输出投影确保维度一致性
    """
    
    def __init__(self, 
                 in_channels=3,
                 expert_configs=None,
                 router_config=None,
                 out_indices=[1, 2, 3],
                 pretrained_configs=None,
                 **kwargs):
        """
        初始化CBDES MoE模型
        
        Args:
            in_channels: 输入图像通道数
            expert_configs: 专家网络配置字典
            router_config: 路由器配置字典
            out_indices: 输出特征图索引
            pretrained_configs: 预训练权重配置
            **kwargs: 其他参数
        """
        super().__init__()
        
        self.out_indices = out_indices
        
        # 默认预训练配置
        if pretrained_configs is None:
            pretrained_configs = {
                'swin': None,  # Swin权重文件路径
                'resnet': 'torchvision',  # 使用torchvision预训练ResNet50
                'convnext': 'torchvision',  # 使用torchvision预训练ConvNeXt
                'pvt': None  # PVT权重文件路径
            }
        
        # Default expert configurations
        if expert_configs is None:
            expert_configs = {
                'swin': {
                    'type': 'SwinTransformer',
                    'embed_dims': 96,
                    'depths': [2, 2, 6, 2],
                    'num_heads': [3, 6, 12, 24],
                    'window_size': 7,
                    'mlp_ratio': 4,
                    'qkv_bias': True,
                    'drop_rate': 0.,
                    'attn_drop_rate': 0.,
                    'drop_path_rate': 0.3,
                    'patch_norm': True,
                    'out_indices': out_indices,
                    'with_cp': False,
                    'convert_weights': True,
                    'pretrained': pretrained_configs['swin']
                },
                'resnet': {
                    'type': 'ResNet',
                    'depth': 50,
                    'num_stages': 4,
                    'out_indices': out_indices,
                    'norm_cfg': {'type': 'BN2d', 'requires_grad': True},
                    'norm_eval': False,
                    'pretrained': pretrained_configs['resnet']
                },
                'convnext': {
                    'type': 'ConvNeXtExpert',
                    'depths': [3, 3, 9, 3],
                    'dims': [96, 192, 384, 768],
                    'drop_path_rate': 0.,
                    'layer_scale_init_value': 1e-6,
                    'out_indices': out_indices,
                    'pretrained': pretrained_configs['convnext']
                },
                'pvt': {
                    'type': 'PyramidVisionTransformerExpert',
                    'img_size': 224,
                    'patch_size': 16,
                    'embed_dims': [64, 128, 320, 512],
                    'num_heads': [1, 2, 5, 8],
                    'mlp_ratios': [8, 8, 4, 4],
                    'qkv_bias': False,
                    'drop_rate': 0.,
                    'attn_drop_rate': 0.,
                    'drop_path_rate': 0.,
                    'depths': [3, 4, 6, 3],
                    'sr_ratios': [8, 4, 2, 1],
                    'out_indices': out_indices,
                    'pretrained': pretrained_configs['pvt']
                }
            }
        
        # Default router configuration
        if router_config is None:
            router_config = {
                'input_dim': 3,  # Input image channels
                'num_experts': 4,
                'embedding_dim': 128,
                'num_heads': 8,
                'dropout': 0.1
            }
        
        # Initialize expert networks
        self.experts = nn.ModuleDict()
        expert_names = ['swin', 'resnet', 'convnext', 'pvt']
        
        for name, config in expert_configs.items():
            # Extract pretrained parameter
            pretrained = config.pop('pretrained', None)
            
            # Remove 'type' field from config as it's not needed for constructor
            config = {k: v for k, v in config.items() if k != 'type'}
            
            if name == 'swin':
                expert = SwinTransformer(**config)
                if pretrained:
                    load_pretrained_weights(expert, pretrained)
                self.experts[name] = expert
            elif name == 'resnet':
                expert = ResNet(**config)
                if pretrained == 'torchvision':
                    # Load torchvision pretrained ResNet50
                    try:
                        pretrained_model = models.resnet50(pretrained=True)
                        # Extract backbone weights (exclude classifier)
                        pretrained_state_dict = {}
                        for name_param, param in pretrained_model.named_parameters():
                            if 'fc' not in name_param:  # Skip final classifier
                                pretrained_state_dict[name_param] = param.data
                        
                        # Load compatible weights
                        model_state_dict = expert.state_dict()
                        for name_param, param in pretrained_state_dict.items():
                            if name_param in model_state_dict and param.shape == model_state_dict[name_param].shape:
                                model_state_dict[name_param] = param
                        
                        expert.load_state_dict(model_state_dict, strict=False)
                        print("Successfully loaded torchvision ResNet50 pretrained weights")
                    except Exception as e:
                        print(f"Failed to load torchvision ResNet50 weights: {e}")
                elif pretrained:
                    load_pretrained_weights(expert, pretrained)
                self.experts[name] = expert
            elif name == 'convnext':
                self.experts[name] = ConvNeXtExpert(**config, pretrained=pretrained)
            elif name == 'pvt':
                self.experts[name] = PyramidVisionTransformerExpert(**config, pretrained=pretrained)
        
        # Initialize router
        self.router = SelfAttentionRouter(**router_config)
        
        # Output projection layers to ensure consistent output dimensions
        self.output_projections = nn.ModuleDict()
        self.target_dim = 256  # Target output dimension for all experts
    
    def init_weights(self):
        """
        初始化权重
        
        由于专家网络已经加载了预训练权重，这个方法主要用于
        初始化路由器和其他新添加的组件。
        """
        # 路由器会使用默认初始化
        for module in self.router.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def forward(self, x):
        """
        CBDES MoE前向传播
        
        这是CBDES MoE的核心前向传播过程，包括：
        1. 通过路由器计算专家权重
        2. 所有专家网络并行处理输入
        3. 动态创建输出投影层确保维度一致
        4. 根据路由器权重加权组合专家输出
        
        Args:
            x: 输入张量 (B, C, H, W)
            
        Returns:
            final_outputs: 多尺度特征图列表
            routing_loss: 负载均衡损失
        """
        # 从路由器获取专家选择权重
        expert_weights, routing_loss = self.router(x)  # (B, num_experts)
        
        # 确保路由权重是有效的（避免全0或NaN）
        if torch.isnan(expert_weights).any() or (expert_weights.sum(dim=-1) < 1e-6).any():
            # 如果路由权重有问题，使用均匀权重
            expert_weights = torch.ones_like(expert_weights) / self.router.num_experts
        
        # 通过所有专家网络进行前向传播
        expert_outputs = {}
        expert_names = list(self.experts.keys())
        
        for name, expert in self.experts.items():
            expert_outputs[name] = expert(x)
        
        # 确保专家输出是列表格式，并且长度正确
        for name in expert_names:
            if not isinstance(expert_outputs[name], (list, tuple)):
                expert_outputs[name] = [expert_outputs[name]]
            # 确保输出数量与out_indices匹配
            if len(expert_outputs[name]) != len(self.out_indices):
                raise ValueError(f"Expert {name} output length {len(expert_outputs[name])} "
                               f"does not match out_indices length {len(self.out_indices)}")
        
        # 动态创建输出投影层（如果不存在）
        for name in expert_names:
            if name not in self.output_projections:
                self.output_projections[name] = nn.ModuleList()
                for i in range(len(self.out_indices)):
                    # 获取专家的实际输出维度
                    actual_dim = expert_outputs[name][i].shape[1]
                    # 仅做通道对齐，保持原始空间尺寸与 vtransform 期望一致
                    projection = nn.Sequential(
                        nn.Conv2d(actual_dim, self.target_dim, 1)
                    )
                    self.output_projections[name].append(projection)
                    # 移动到与专家输出相同的设备
                    projection.to(expert_outputs[name][i].device)
        
        # 应用输出投影确保通道一致（空间尺寸先不变）
        projected_outputs = {}
        for name in expert_names:
            projected_outputs[name] = []
            for i in range(len(self.out_indices)):
                projected_feat = self.output_projections[name][i](expert_outputs[name][i])
                projected_outputs[name].append(projected_feat)
        
        # 根据路由器权重加权组合专家输出
        final_outputs = []
        for i in range(len(self.out_indices)):
            # 对第 i 个尺度，先对齐所有专家的空间分辨率到统一目标
            # 选择第一个专家的尺寸作为目标，或选择最大尺寸
            target_H, target_W = None, None
            for j, name in enumerate(expert_names):
                H, W = projected_outputs[name][i].shape[-2:]
                if target_H is None:
                    target_H, target_W = H, W
                else:
                    if H * W > target_H * target_W:
                        target_H, target_W = H, W

            # 进行空间对齐
            aligned_feats = []
            for j, name in enumerate(expert_names):
                feat = projected_outputs[name][i]
                if feat.shape[-2:] != (target_H, target_W):
                    feat = F.interpolate(feat, size=(target_H, target_W), mode='bilinear', align_corners=False)
                aligned_feats.append(feat)

            weighted_features = None
            for j, name in enumerate(expert_names):
                expert_feat = aligned_feats[j]
                # 扩展权重维度以匹配特征图维度
                weight = expert_weights[:, j:j+1].unsqueeze(-1).unsqueeze(-1)
                
                if weighted_features is None:
                    weighted_features = weight * expert_feat
                else:
                    weighted_features += weight * expert_feat
            
            # 确保加权后的特征不是NaN或Inf
            if torch.isnan(weighted_features).any() or torch.isinf(weighted_features).any():
                # 如果特征有问题，使用第一个专家的特征
                weighted_features = aligned_feats[0]
            
            final_outputs.append(weighted_features)
        
        # 保存路由损失供上层模型读取（如需要）
        self.routing_loss = routing_loss
        # 返回多尺度特征和路由损失，满足CBDESBEVFusion的输入约定
        return final_outputs, routing_loss
    
    def get_expert_utilization(self):
        """
        获取专家利用率统计信息
        
        返回每个专家网络的使用情况统计，用于监控负载均衡效果
        和模型训练状态分析。
        
        Returns:
            dict: 专家利用率字典，键为'expert_i'，值为利用率比例
        """
        if hasattr(self.router, 'expert_counts'):
            total = self.router.total_samples.item()
            if total > 0:
                utilization = self.router.expert_counts / total
                return {f'expert_{i}': utilization[i].item() for i in range(len(utilization))}
        return {}
