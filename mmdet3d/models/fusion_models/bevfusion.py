"""
BEVFusion 融合模型

这个文件实现了BEVFusion的核心架构，用于BEV（鸟瞰图）空间的多模态感知。

主要功能：
1. 多模态编码器：处理相机、LiDAR、雷达等不同传感器数据
2. 特征提取：从每个传感器中提取高级语义特征
3. 视图变换：将图像特征从图像空间转换到BEV空间
4. 特征融合：将多模态特征进行融合
5. 解码器：生成最终的BEV特征图
6. 任务头：支持3D检测和BEV分割任务

架构流程：
输入图像/点云/雷达 → 编码器 → 视图变换 → 融合器 → 解码器 → 检测/分割头
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

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    """
    BEVFusion: 多模态BEV感知模型
    
    这是BEVFusion的主要模型类，集成了相机、LiDAR和雷达三种传感器数据。
    
    关键特性：
    - 支持多种传感器模态（相机、LiDAR、雷达）
    - 自动视图变换（图像特征 → BEV空间）
    - 可选的跨模态特征融合
    - 统一的BEV空间特征表示
    - 支持3D目标检测和BEV地图分割
    
    Args:
        encoders (Dict): 各传感器的编码器配置
            - camera: 相机编码器（backbone + neck + vtransform）
            - lidar: LiDAR编码器（voxelize + backbone）
            - radar: 雷达编码器（voxelize + backbone）
        fuser (Dict, optional): 多模态融合器配置
        decoder (Dict): 解码器配置（backbone + neck）
        heads (Dict): 任务头配置（object检测 / map分割）
        **kwargs: 其他配置参数
    """
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        # ========== 1. 初始化各传感器编码器 ==========
        self.encoders = nn.ModuleDict()
        
        # 相机编码器：包含backbone、neck和视图变换模块
        # backbone: 图像特征提取（如SwinTransformer、CBDESMoE）
        # neck: 多尺度特征融合（如GeneralizedLSSFPN）
        # vtransform: 图像空间→BEV空间的视图变换（如LSSTransform）
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        
        # LiDAR编码器：点云体素化 + 特征提取
        # voxelize: 将3D点云转换为稀疏体素特征
        # backbone: 稀疏卷积网络提取点云特征
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                # 硬体素化：限制每个体素的点数
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                # 动态散射：可变点数体素化
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        # 雷达编码器：与LiDAR类似，但处理雷达点云数据
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

        # ========== 2. 初始化融合器 ==========
        # 融合器负责将来自不同传感器的特征进行融合
        # 例如：ConvFuser融合相机和LiDAR特征
        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        # ========== 3. 初始化解码器 ==========
        # 解码器对融合后的BEV特征进行进一步处理
        # backbone: ResNet等标准网络进行特征提取
        # neck: FPN等结构进行多尺度特征融合
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        
        # ========== 4. 初始化任务头 ==========
        # object: 3D目标检测头
        # map: BEV地图分割头
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        # ========== 5. 配置损失权重 ==========
        # 用于平衡不同任务的损失大小
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # ========== 6. 深度监督损失配置 ==========
        # 如果使用BEVDepth等视图变换方法，需要深度监督损失
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

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
    ) -> torch.Tensor:
        """
        提取相机图像特征并转换到BEV空间
        
        这是BEVFusion的核心功能之一：将多视角图像特征提取并转换到统一的BEV空间。
        
        处理流程：
        1. 图像特征提取：通过backbone（如SwinTransformer）提取多尺度特征
        2. 特征融合：通过neck进行多尺度特征融合
        3. 视图变换：将图像特征从图像空间变换到BEV空间
        4. 深度估计：可选地使用深度监督损失
        
        Args:
            x: 输入图像张量 (B, N, C, H, W)
               B: batch size
               N: 相机数量（通常为6）
               C: 图像通道数
               H, W: 图像高度和宽度
            points: LiDAR点云数据
            radar_points: 雷达点云数据
            camera2ego: 相机到ego车身的变换矩阵
            lidar2ego: LiDAR到ego车身的变换矩阵
            lidar2camera: LiDAR到相机的变换矩阵
            lidar2image: LiDAR到图像的投影矩阵
            camera_intrinsics: 相机内参
            camera2lidar: 相机到LiDAR的变换矩阵
            img_aug_matrix: 图像数据增强矩阵
            lidar_aug_matrix: LiDAR数据增强矩阵
            img_metas: 图像元信息
            gt_depths: 真实深度图（用于深度监督）
        
        Returns:
            BEV空间的特征张量
        """
        # ========== Step 1: 预处理输入图像 ==========
        B, N, C, H, W = x.size()
        # 将多个相机的图像展平：(B, N, C, H, W) -> (B*N, C, H, W)
        x = x.view(B * N, C, H, W)

        # ========== Step 2: 图像特征提取 ==========
        # 使用backbone提取多尺度图像特征
        # 例如：SwinTransformer输出 [192, 384, 768] 三个尺度的特征
        x = self.encoders["camera"]["backbone"](x)
        
        # ========== Step 3: 特征融合 ==========
        # 使用neck（如GeneralizedLSSFPN）融合多尺度特征
        # 输出统一尺寸的特征图 (B*N, 256, H', W')
        x = self.encoders["camera"]["neck"](x)

        # 如果neck返回的是列表，取第一个元素
        if not isinstance(x, torch.Tensor):
            x = x[0]

        # ========== Step 4: 恢复批次和相机维度 ==========
        BN, C, H, W = x.size()
        # 从展平的张量恢复为多相机格式：(B*N, C, H, W) -> (B, N, C, H, W)
        x = x.view(B, int(BN / B), C, H, W)

        # ========== Step 5: 视图变换（图像空间 → BEV空间）==========
        # 这是BEVFusion的关键步骤：使用Lift-Splat-Shoot (LSS) 将图像特征投影到BEV空间
        # LSSTransform的工作原理：
        # 1. Lift: 为每个像素预测深度分布
        # 2. Splat: 基于相机内参和外参将特征投影到3D空间
        # 3. Shoot: 将3D特征栅格化到BEV网格
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
        return x  # 返回BEV空间的特征张量
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        """
        提取LiDAR或雷达的点云特征
        
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
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        """
        将点云体素化（Voxelization）
        
        体素化的作用：
        1. 将无序的点云转换为规则的稀疏体素网格
        2. 每个体素包含多个点的特征聚合
        3. 保留体素的3D坐标信息用于稀疏卷积
        
        有两种体素化方式：
        - 硬体素化（Hard Voxelize）：限制每个体素的最大点数，返回size信息
        - 动态散射（Dynamic Scatter）：不限制点数，自适应聚合
        
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

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

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
        # 目前只支持单样本输入，不支持批量列表
        if isinstance(img, list):
            raise NotImplementedError
        else:
            # 调用单样本前向传播
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
        单样本的前向传播
        
        这是BEVFusion的核心前向传播流程：
        
        1. 多模态特征提取（相机/LiDAR/雷达）
        2. 特征融合（可选）
        3. 解码器处理
        4. 任务头输出
        
        Args:
            img: 多视角图像 (B, N, C, H, W)
            points: LiDAR点云
            radar: 雷达点云
            camera2ego, lidar2ego等: 各种变换矩阵
            depths: 深度监督
            gt_masks_bev: BEV分割的ground truth
            gt_bboxes_3d: 3D检测的ground truth
            gt_labels_3d: 3D检测的类别标签
        
        Returns:
            training: 损失字典
            inference: 预测结果
        """
        
        # ========== 阶段1: 多模态特征提取 ==========
        features = []
        auxiliary_losses = {}
        
        # 传感器处理顺序：
        # 训练时：按定义顺序（通常是相机 → LiDAR → 雷达）
        # 推理时：反转顺序（避免OOM，先处理轻量的）
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # 相机特征提取：图像 → BEV特征
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
            return outputs
        else:
            # ========== 推理模式：生成预测结果 ==========
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    # 3D目标检测：生成检测框
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    # 将结果移动到CPU并添加到输出
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
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

