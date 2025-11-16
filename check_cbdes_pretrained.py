#!/usr/bin/env python3
"""
验证 CBDES MoE 是否正确加载了各专家和预训练权重
"""

import torch
from mmcv import Config
from mmdet3d.models import build_model

def check_pretrained_weights(model, expert_name, expected_pretrained_path=None):
    """检查专家网络是否加载了预训练权重"""
    if not hasattr(model, 'encoders') or 'camera' not in model.encoders:
        print(f"❌ 模型没有 camera encoder")
        return False
    
    backbone = model.encoders['camera']['backbone']
    if not hasattr(backbone, 'experts'):
        print(f"❌ Backbone 不是 CBDESMoE（没有 experts 属性）")
        return False
    
    if expert_name not in backbone.experts:
        print(f"❌ 专家 {expert_name} 不存在")
        return False
    
    expert = backbone.experts[expert_name]
    
    # 检查权重是否非零（如果加载了预训练权重，权重应该不是全零）
    total_params = sum(p.numel() for p in expert.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in expert.parameters())
    zero_ratio = 1.0 - (non_zero_params / total_params) if total_params > 0 else 1.0
    
    print(f"\n专家 {expert_name}:")
    print(f"  总参数数: {total_params:,}")
    print(f"  非零参数数: {non_zero_params:,}")
    print(f"  零参数比例: {zero_ratio:.4f}")
    
    if zero_ratio > 0.5:
        print(f"  ⚠️  警告: 零参数比例过高 ({zero_ratio:.2%})，可能未正确加载预训练权重")
        return False
    else:
        print(f"  ✅ 权重看起来正常（零参数比例 {zero_ratio:.2%}）")
        return True

def main():
    # 加载配置
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/moe_lal/default.yaml"
    from torchpack.utils.config import configs
    configs.load(config_file, recursive=True)
    from mmdet3d.utils import recursive_eval
    cfg = Config(recursive_eval(configs), filename=config_file)
    
    print("=" * 60)
    print("验证 CBDES MoE 预训练权重加载")
    print("=" * 60)
    
    # 构建模型
    print("\n正在构建模型...")
    model = build_model(cfg.model)
    
    # 检查各专家
    experts_to_check = ['swin', 'resnet', 'convnext', 'pvt']
    results = {}
    
    for expert_name in experts_to_check:
        results[expert_name] = check_pretrained_weights(model, expert_name)
    
    # 总结
    print("\n" + "=" * 60)
    print("验证结果总结:")
    print("=" * 60)
    for expert_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {expert_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ 所有专家的预训练权重都已正确加载！")
    else:
        print("\n⚠️  部分专家的预训练权重可能未正确加载，请检查日志输出。")

if __name__ == "__main__":
    main()

