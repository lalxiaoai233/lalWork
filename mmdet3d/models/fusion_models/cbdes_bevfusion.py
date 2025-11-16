"""
CBDES BEVFusion: 集成CBDES MoE的BEVFusion模型

这个模块扩展了原始BEVFusion，使用CBDES MoE（分层解耦专家混合）架构
来增强场景适应性和特征表示能力。

与原始BEVFusion的主要区别：
1. 相机backbone替换：使用CBDES MoE替代传统单backbone（如SwinTransformer）
2. 动态专家选择：根据输入自动选择最合适的专家网络
3. 路由损失：添加负载均衡损失，确保专家均衡使用
4. 专家利用统计：支持监控各专家的使用情况

核心改进：
- 使用4个异构专家网络（Swin、ResNet、ConvNeXt、PVT）
- 自注意力路由器进行智能专家选择
- 自动负载均衡确保训练稳定性
"""

from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

__all__ = ["CBDESBEVFusion"]


@FUSIONMODELS.register_module()
class CBDESBEVFusion(Base3DFusionModel):
    """
    CBDES BEVFusion: 集成CBDES MoE的BEVFusion模型
    
    这是BEVFusion的增强版本，在相机编码器中集成了CBDES MoE（分层解耦专家混合）架构。
    
    与原始BEVFusion的对比：
    
    | 特性 | BEVFusion | CBDESBEVFusion |
    |------|-----------|----------------|
    | 相机backbone | 单模型（如SwinTransformer） | CBDES MoE（4个异构专家） |
    | 特征提取 | 固定backbone | 动态专家选择 |
    | 路由机制 | 无 | 自注意力路由器 |
    | 损失函数 | 任务损失 | 任务损失 + 路由损失 |
    | 专家监控 | 无 | 支持专家利用率统计 |
    
    核心优势：
    1. 更强的特征表示能力：4个异构专家的多样化特征
    2. 自适应场景：根据输入动态选择最合适的专家
    3. 负载均衡：通过路由损失确保所有专家都被利用
    4. 向后兼容：可以关闭CBDES MoE，回退到原始BEVFusion行为
    
    Args:
        encoders: 编码器配置（与BEVFusion相同）
        fuser: 融合器配置（与BEVFusion相同）
        decoder: 解码器配置（与BEVFusion相同）
        heads: 任务头配置（与BEVFusion相同）
        use_cbdes_moe: 是否启用CBDES MoE（默认True）
        cbdes_moe_config: CBDES MoE的配置参数
            - expert_configs: 各专家网络的配置
            - router_config: 路由器的配置
        **kwargs: 其他配置参数
    """
    
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        use_cbdes_moe: bool = True,
        cbdes_moe_config: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # ========== CBDESBEVFusion 专用配置 ==========
        # 用于控制是否启用CBDES MoE（可以随时切换回原始BEVFusion行为）
        self.use_cbdes_moe = use_cbdes_moe
        
        # ========== 1. 初始化编码器 ==========
        # 注意：与BEVFusion的主要区别在这里 - 相机backbone会被替换为CBDESMoE
        self.encoders = nn.ModuleDict()
        
        # ========== 【关键差异点1】相机编码器：替换为CBDES MoE ==========
        # BEVFusion: 使用单个backbone（如SwinTransformer、ResNet等）
        # CBDESBEVFusion: 使用CBDES MoE（包含4个异构专家网络 + 自注意力路由器）
        if encoders.get("camera") is not None:
            camera_config = encoders["camera"].copy()
            
            if self.use_cbdes_moe:
                # 【核心修改】将原来的单个backbone替换为CBDES MoE
                # 如果配置文件中已经指定了CBDESMoE，直接使用配置文件中的配置
                # 否则，使用cbdes_moe_config或创建默认配置
                existing_backbone = camera_config.get("backbone", {})
                if existing_backbone.get("type") == "CBDESMoE":
                    # 配置文件中已经指定了CBDESMoE，直接使用
                    camera_config["backbone"] = existing_backbone
                else:
                    # 配置文件中是其他backbone，替换为CBDES MoE
                    camera_config["backbone"] = {
                        "type": "CBDESMoE",  # 使用CBDES MoE作为backbone
                        "in_channels": 3,
                        "out_indices": existing_backbone.get("out_indices", [1, 2, 3]),
                        # 合并CBDES MoE的完整配置（专家配置 + 路由器配置）
                        **(cbdes_moe_config or {})
                    }
            
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(camera_config["backbone"]),  # 可能是CBDES MoE
                    "neck": build_neck(camera_config["neck"]),  # 与BEVFusion相同
                    "vtransform": build_vtransform(camera_config["vtransform"]),  # 与BEVFusion相同
                }
            )
        
        # ========== 2. LiDAR编码器（与BEVFusion完全相同）==========
        # 注意：LiDAR和Radar编码器没有使用CBDES MoE，保持原样
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # ========== 3. 雷达编码器（与BEVFusion完全相同）==========
        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        # ========== 4. 融合器（与BEVFusion完全相同）==========
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        # ========== 5. 解码器（与BEVFusion完全相同）==========
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        # ========== 6. 任务头（与BEVFusion完全相同）==========
        self.heads = nn.ModuleDict()
        for name, cfg in heads.items():
            if cfg is not None:
                self.heads[name] = build_head(cfg)

        # ========== 7. 损失配置（与BEVFusion完全相同）==========
        # 配置损失权重
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # ========== 8. 深度监督损失配置（与BEVFusion完全相同）==========
        # 如果使用BEVDepth等视图变换方法，需要深度监督损失
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        # ========== 【关键差异点2】路由损失累积 ==========
        # BEVFusion: 没有路由损失
        # CBDESBEVFusion: 累积路由损失用于训练时的负载均衡
        # 这是一个列表，存储每个batch的路由损失
        self.routing_losses = []
        
        # 初始化模型权重
        self.init_weights()
    
    def init_weights(self) -> None:
        """初始化模型权重，通常加载预训练权重"""
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
        
    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ):
        """
        【核心差异】提取相机特征 - 使用CBDES MoE
        
        这是CBDESBEVFusion与BEVFusion最重要的方法差异！
        
        BEVFusion的extract_camera_features:
        1. 输入图像 → backbone（单一模型，如SwinTransformer）→ 多尺度特征
        2. neck融合 → 视图变换 → BEV特征
        3. 返回：BEV特征张量
        
        CBDESBEVFusion的extract_camera_features:
        1. 输入图像 → CBDES MoE（4个异构专家 + 路由器）
        2. 路由器计算专家权重，专家网络处理输入
        3. 加权组合专家输出 → neck → 视图变换 → BEV特征
        4. 返回：BEV特征张量 + 路由损失（用于负载均衡）
        
        差异总结：
        - BEVFusion: 固定使用单一backbone
        - CBDESBEVFusion: 动态使用4个专家，根据输入自动选择最优组合
        
        Args:
            img: 输入图像 (B, N, C, H, W)
            points, radar等: 其他参数与BEVFusion相同
            gt_depths: 深度监督（可选）
        
        Returns:
            如果启用深度监督：features, depth_loss
            否则：features
        """
        # 检查backbone是否是CBDESMoE类型（通过检查是否有router属性）
        # CBDESMoE有router属性，而其他backbone（如SwinTransformer）没有
        is_cbdes_moe_backbone = hasattr(self.encoders["camera"]["backbone"], "router")
        
        if self.use_cbdes_moe and is_cbdes_moe_backbone:
            # ========== Step 1: 预处理输入图像（与BEVFusion相同）==========
            B, N, C, H, W = x.size()
            # 将多个相机的图像展平：(B, N, C, H, W) -> (B*N, C, H, W)
            x = x.view(B * N, C, H, W)
            
            # ========== 【关键步骤1】CBDES MoE前向传播 ==========
            # BEVFusion: backbone返回单一特征
            # CBDESBEVFusion: backbone返回（特征, 路由损失）
            # 这里使用4个异构专家网络进行特征提取，并通过路由器动态选择
            features, routing_loss = self.encoders["camera"]["backbone"](x)
            
            # ========== 【关键步骤2】累积路由损失 ==========
            # BEVFusion: 没有路由损失
            # CBDESBEVFusion: 收集路由损失用于训练时的负载均衡
            # 路由损失确保所有专家都被使用，防止某些专家被忽略
            if self.training:
                self.routing_losses.append(routing_loss)
            
            # ========== Step 3: 特征融合（与BEVFusion相同）==========
            # 使用neck（如GeneralizedLSSFPN）融合多尺度特征
            features = self.encoders["camera"]["neck"](features)
            
            # 如果neck返回的是列表，取第一个元素（与BEVFusion相同）
            if not isinstance(features, torch.Tensor):
                features = features[0]
            
            # ========== Step 4: 恢复批次和相机维度（与BEVFusion相同）==========
            BN, C, H, W = features.size()
            # 从展平的张量恢复为多相机格式：(B*N, C, H, W) -> (B, N, C, H, W)
            features = features.view(B, int(BN / B), C, H, W)
            
            # ========== Step 5: 视图变换（与BEVFusion相同）==========
            # 将图像特征从图像空间变换到BEV空间
            if self.use_depth_loss and gt_depths is not None:
                features, depth_loss = self.encoders["camera"]["vtransform"](
                    features,
                    points,
                    radar_points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    img_metas,
                    depth_loss=self.use_depth_loss,
                    gt_depths=gt_depths,
                )
                return features, depth_loss
            else:
                features = self.encoders["camera"]["vtransform"](
                    features,
                    points,
                    radar_points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    img_metas,
                )
                return features
        else:
            # ========== 回退模式 ==========
            # 如果未启用CBDES MoE或backbone不是CBDESMoE，使用标准backbone处理
            # 这与BEVFusion的逻辑完全相同
            B, N, C, H, W = x.size()
            x = x.view(B * N, C, H, W)
            
            # 使用标准backbone提取特征（不是CBDES MoE）
            x = self.encoders["camera"]["backbone"](x)
            
            # 使用neck融合多尺度特征
            x = self.encoders["camera"]["neck"](x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            
            # 恢复批次和相机维度
            BN, C, H, W = x.size()
            x = x.view(B, int(BN / B), C, H, W)
            
            # 视图变换
            x = self.encoders["camera"]["vtransform"](
                x,
                points,
                radar_points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                depth_loss=self.use_depth_loss,
                gt_depths=gt_depths,
            )
            return x
    
    @force_fp32()
    def voxelize(self, points, sensor):
        """
        将点云体素化（Voxelization）
        
        与BEVFusion完全相同，因为体素化与CBDES MoE无关。
        
        体素化的作用：
        1. 将无序的点云转换为规则的稀疏体素网格
        2. 每个体素包含多个点的特征聚合
        3. 保留体素的3D坐标信息用于稀疏卷积
        
        Args:
            points: 点云列表（每个元素代表一个sweep的点云）
            sensor: 传感器类型（"lidar" 或 "radar"）
        
        Returns:
            feats: 体素特征 (M, C) - M个非空体素，C维特征
            coords: 体素坐标 (M, 4) - [batch_idx, x, y, z]
            sizes: 每个体素的点数（如果使用硬体素化）
        """
        feats, coords, sizes = [], [], []
        
        # 对每个sweep的点云进行体素化
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            
            # 硬体素化返回 (feats, coords, sizes)
            if len(ret) == 3:
                f, c, n = ret
            # 动态散射返回 (feats, coords)
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            
            feats.append(f)
            # 添加batch维度：coords从 [N, 3] -> [N, 4] (添加batch_idx=k)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        # 拼接所有sweep的体素特征
        feats = torch.cat(feats, dim=0)  # (M, C)
        coords = torch.cat(coords, dim=0)  # (M, 4)
        
        # 如果有size信息（硬体素化），计算平均值
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                # 平均聚合：对同一体素内的多个点求平均
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        """
        提取LiDAR或雷达的点云特征
        
        与BEVFusion完全相同，因为LiDAR/雷达特征提取与CBDES MoE无关。
        
        Args:
            x: 点云数据
            sensor: 传感器类型（"lidar" 或 "radar"）
        
        Returns:
            点云特征张量
        """
        # 步骤1：体素化 - 将点云转换为体素特征
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        
        # 步骤2：使用稀疏卷积网络提取特征
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @auto_fp16(apply_to=("img", "points"))  # 自动使用FP16混合精度训练
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        BEVFusion的前向传播入口
        
        这是模型的顶层前向传播方法，处理批量数据和单样本数据。
        目前只支持单样本处理，批量列表模式未实现。
        
        装饰器说明：
        - @auto_fp16: 自动使用FP16混合精度，加速训练并节省显存
        - apply_to=("img", "points"): 只对图像和点云使用FP16，其他部分保持FP32
        
        Returns:
            forward_single的返回值（损失或预测）
        """
        # ========== 【关键步骤】清空路由损失列表 ==========
        # 每个前向传播开始时重置路由损失，避免累积
        # 路由损失会在forward_single中收集并添加到输出
        self.routing_losses = []
        
        # 目前只支持单样本输入，不支持批量列表
        if isinstance(img, list):
            raise NotImplementedError
        else:
            # 调用单样本前向传播
            # 路由损失会在forward_single中自动添加到输出字典
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        """
        单样本前向传播（基本与BEVFusion相同）
        
        注意：这个方法主要用于处理辅助损失和路由损失。
        大部分逻辑与BEVFusion相同，只是在训练时收集额外的路由损失。
        
        【主要区别】：
        - BEVFusion: 输出损失字典
        - CBDESBEVFusion: 输出损失字典 + 路由损失（如果启用CBDES MoE）
        """
        # ========== 阶段1: 多模态特征提取 ==========
        features = []
        auxiliary_losses = {}
        
        # 传感器处理顺序（与BEVFusion完全相同）：
        # 训练时：按定义顺序（通常是相机 → LiDAR → 雷达）
        # 推理时：反转顺序（避免OOM，先处理轻量的）
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # 相机特征提取：图像 → BEV特征
                # 【关键差异】这里调用重写的extract_camera_features
                # 会使用CBDES MoE并收集路由损失
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                # 如果使用深度监督，提取辅助损失
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                # LiDAR特征提取：点云 → BEV特征
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                # 雷达特征提取：雷达点云 → BEV特征
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        # 推理时反转特征顺序以避免OOM
        if not self.training:
            features = features[::-1]

        # ========== 阶段2: 特征融合 ==========
        # 如果配置了融合器（如ConvFuser），将多模态特征融合
        # 否则直接使用单个模态的特征
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        # ========== 阶段3: 解码器处理 ==========
        # 解码器对融合后的BEV特征进行进一步处理
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        # ========== 阶段4: 任务头输出 ==========
        if self.training:
            # ========== 训练模式：计算损失 ==========
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    # 3D目标检测：生成预测框并计算检测损失
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    # BEV地图分割：计算分割损失
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                
                # 保存损失到输出字典
                for name, val in losses.items():
                    if val.requires_grad:
                        # 可反向传播的损失（需要乘以权重）
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        # 统计指标（不需要梯度）
                        outputs[f"stats/{type}/{name}"] = val
            
            # 如果有深度监督，添加深度损失
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            
            # ========== 【关键差异】添加路由损失 ==========
            # BEVFusion: 没有这一步
            # CBDESBEVFusion: 添加路由损失用于负载均衡
            # 路由损失确保所有专家都被使用，防止某些专家被忽略（专家塌陷）
            if self.routing_losses:
                # 计算平均路由损失（可能包含多个batch的损失）
                routing_loss = sum(self.routing_losses) / len(self.routing_losses)
                outputs['routing_loss'] = routing_loss
            
            return outputs
        else:
            # ========== 推理模式：生成预测结果（与BEVFusion相同）==========
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    # 3D目标检测：生成检测框
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    # 将结果移动到CPU并添加到输出
                    # get_bboxes返回[[boxes, scores, labels]]，外层是layers，内层是batches
                    # 与BEVFusion完全一致：直接解包bboxes中的每个元素
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    # BEV地图分割：生成分割结果
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu() if gt_masks_bev is not None else None,
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def get_expert_utilization(self):
        """
        【CBDESBEVFusion 独有功能】获取专家利用率统计
        
        这是CBDESBEVFusion新增的功能，用于监控训练过程中各专家的使用情况。
        BEVFusion没有此功能。
        
        专家利用率统计的作用：
        1. 监控训练：检查所有专家是否被均衡使用
        2. 调试路由：如果某个专家利用率很低，可能需要调整路由策略
        3. 性能优化：识别未充分利用的专家，优化模型架构
        
        Returns:
            dict: 专家利用率字典，键为'expert_i'，值为利用率比例
                  例如: {'expert_0': 0.25, 'expert_1': 0.30, 'expert_2': 0.20, 'expert_3': 0.25}
                  表示expert_1使用最多，expert_2使用最少
        
        Example:
            >>> model = CBDESBEVFusion(...)
            >>> utilization = model.get_expert_utilization()
            >>> print(utilization)
            {'expert_0': 0.28, 'expert_1': 0.25, 'expert_2': 0.22, 'expert_3': 0.25}
        """
        if self.use_cbdes_moe and hasattr(self.encoders["camera"]["backbone"], "get_expert_utilization"):
            # 从CBDES MoE backbone获取专家利用率统计
            return self.encoders["camera"]["backbone"].get_expert_utilization()
        return {}  # 如果未启用CBDES MoE或没有此方法，返回空字典
