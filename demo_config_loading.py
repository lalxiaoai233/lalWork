#!/usr/bin/env python3
"""
演示配置文件加载和合并机制的脚本

展示：
1. torchpack.configs.load() 的 recursive 模式
2. mmcv.Config 的 _base_ 继承
3. 配置合并的优先级
4. 配置如何传递到模型
"""

import os
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval

def demo_config_loading():
    """演示配置加载的完整流程"""
    
    print("=" * 80)
    print("配置文件加载机制演示")
    print("=" * 80)
    
    # 配置文件路径
    config_file = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/moe_lal/convfuser.yaml"
    
    print(f"\n📁 加载配置文件: {config_file}")
    print("-" * 80)
    
    # ========== 步骤1: torchpack 递归加载 ==========
    print("\n【步骤1】torchpack.configs.load(recursive=True)")
    print("  功能: 递归向上查找并加载父目录的 default.yaml")
    
    # 清空配置
    configs.clear()
    
    # 模拟递归加载过程
    base_dir = os.path.dirname(config_file)
    parent_defaults = []
    current_dir = base_dir
    while current_dir and current_dir != os.path.dirname(current_dir):
        default_path = os.path.join(current_dir, "default.yaml")
        if os.path.exists(default_path):
            parent_defaults.append(default_path)
        current_dir = os.path.dirname(current_dir)
    
    print(f"  找到的父目录 default.yaml:")
    for i, path in enumerate(parent_defaults, 1):
        print(f"    {i}. {path}")
    
    # 实际加载
    configs.load(config_file, recursive=True)
    print(f"  ✅ 配置已加载到 torchpack.configs")
    
    # ========== 步骤2: 显示 _base_ 字段 ==========
    print("\n【步骤2】检查 _base_ 字段")
    if '_base_' in configs:
        print(f"  _base_: {configs['_base_']}")
        print("  mmcv.Config 会加载 _base_ 中指定的配置文件并合并")
    else:
        print("  当前配置没有 _base_ 字段")
    
    # ========== 步骤3: 转换为 mmcv.Config ==========
    print("\n【步骤3】转换为 mmcv.Config")
    print("  功能: 处理 _base_ 继承、变量替换(${...})、配置合并")
    
    cfg = Config(recursive_eval(configs), filename=config_file)
    print(f"  ✅ 已转换为 mmcv.Config 对象")
    
    # ========== 步骤4: 显示关键配置值 ==========
    print("\n【步骤4】关键配置值")
    print("-" * 80)
    
    if 'model' in cfg:
        print(f"  model.type: {cfg.model.get('type', 'NOT SET')}")
        print(f"  model.use_cbdes_moe: {cfg.model.get('use_cbdes_moe', 'NOT SET')}")
        
        if 'encoders' in cfg.model and 'camera' in cfg.model.encoders:
            camera_backbone = cfg.model.encoders.camera.get('backbone', {})
            print(f"  model.encoders.camera.backbone.type: {camera_backbone.get('type', 'NOT SET')}")
        
        if 'cbdes_moe_config' in cfg.model:
            expert_configs = cfg.model.cbdes_moe_config.get('expert_configs', {})
            print(f"  model.cbdes_moe_config.expert_configs: {list(expert_configs.keys())}")
    
    # ========== 步骤5: 演示命令行参数覆盖 ==========
    print("\n【步骤5】命令行参数覆盖（模拟）")
    print("  功能: 命令行参数具有最高优先级")
    
    # 模拟命令行参数
    test_opts = ['--model.use_cbdes_moe=true']
    print(f"  模拟命令行参数: {test_opts}")
    
    configs.update(test_opts)
    cfg = Config(recursive_eval(configs), filename=config_file)
    print(f"  ✅ 覆盖后的 model.use_cbdes_moe: {cfg.model.get('use_cbdes_moe', 'NOT SET')}")
    
    # ========== 步骤6: 配置传递到模型 ==========
    print("\n【步骤6】配置传递到模型")
    print("  功能: build_model(cfg.model) 将配置字典作为关键字参数传递")
    print("-" * 80)
    
    print("  build_model(cfg.model) 内部执行:")
    print("    1. 从 cfg.model 提取 'type': 'CBDESBEVFusion'")
    print("    2. 调用 CBDESBEVFusion(**cfg.model)")
    print("    3. 等价于:")
    print("       CBDESBEVFusion(")
    print("           encoders=cfg.model['encoders'],")
    print("           fuser=cfg.model['fuser'],")
    print("           decoder=cfg.model['decoder'],")
    print("           heads=cfg.model['heads'],")
    print("           use_cbdes_moe=cfg.model['use_cbdes_moe'],  # 从配置文件读取")
    print("           cbdes_moe_config=cfg.model['cbdes_moe_config'],")
    print("           ...")
    print("       )")
    
    # ========== 步骤7: 配置合并优先级总结 ==========
    print("\n" + "=" * 80)
    print("配置合并优先级（从低到高）")
    print("=" * 80)
    print("""
    1. 【最低】torchpack recursive 加载的父目录 default.yaml
       - 例如: camera+lidar/default.yaml
       
    2. 当前配置文件的内容
       - 例如: moe_lal/default.yaml
       
    3. _base_ 中指定的配置文件（按顺序，后加载的覆盖先加载的）
       - 例如: convfuser.yaml 中的 _base_: ['default.yaml']
       
    4. 【最高】命令行参数
       - 例如: --model.use_cbdes_moe=true
    """)
    
    # ========== 步骤8: 实际配置来源追踪 ==========
    print("\n" + "=" * 80)
    print("实际配置来源追踪")
    print("=" * 80)
    
    print(f"\n当前配置文件中 use_cbdes_moe 的值来源:")
    print(f"  1. 检查 moe_lal/default.yaml")
    default_yaml = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/moe_lal/default.yaml"
    if os.path.exists(default_yaml):
        with open(default_yaml, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if 'use_cbdes_moe' in line:
                    print(f"     第 {line_num} 行: {line.strip()}")
                    break
    
    print(f"  2. 检查 convfuser.yaml")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if 'use_cbdes_moe' in line:
                    print(f"     第 {line_num} 行: {line.strip()}")
                    break
    
    print(f"  3. 最终值: {cfg.model.get('use_cbdes_moe', 'NOT SET')}")
    
    print("\n" + "=" * 80)
    print("演示完成")
    print("=" * 80)

if __name__ == "__main__":
    demo_config_loading()

