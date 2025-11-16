# 配置文件加载流程图

## 完整调用链

```
训练命令
  ↓
python tools/train_cbdes_moe.py configs/.../convfuser.yaml --model.use_cbdes_moe=true
  ↓
【步骤1】torchpack.configs.load(convfuser.yaml, recursive=True)
  │
  ├─ 加载 convfuser.yaml
  │   └─ 内容: _base_: ['default.yaml'], model.fuser: {...}
  │
  ├─ 递归向上查找父目录的 default.yaml
  │   ├─ moe_lal/default.yaml          ← 找到
  │   ├─ camera+lidar/default.yaml     ← 找到
  │   ├─ secfpn/default.yaml           ← 可能找到
  │   └─ ... (继续向上查找)
  │
  └─ 按顺序合并（后加载的覆盖先加载的）
      └─ 结果存储在 torchpack.configs 对象中

  ↓
【步骤2】configs.update(opts)
  │
  └─ 处理命令行参数
      └─ --model.use_cbdes_moe=true → configs['model']['use_cbdes_moe'] = True

  ↓
【步骤3】Config(recursive_eval(configs), filename=...)
  │
  ├─ recursive_eval(configs)
  │   └─ 处理 ${...} 变量替换
  │       └─ 例如: ${image_size} → [256, 704]
  │
  ├─ mmcv.Config.__init__()
  │   ├─ 读取 _base_ 字段
  │   ├─ 递归加载 _base_ 中指定的配置文件
  │   │   └─ 相对于当前文件路径解析
  │   │       └─ convfuser.yaml 的 _base_: ['default.yaml']
  │   │           → 加载 moe_lal/default.yaml
  │   │
  │   └─ 合并配置（后加载的覆盖先加载的）
  │       └─ 最终配置存储在 cfg 对象中
  │
  └─ 返回 mmcv.Config 对象

  ↓
【步骤4】build_model(cfg.model)
  │
  ├─ 从 cfg.model 提取 'type': 'CBDESBEVFusion'
  ├─ 查找注册的模型类
  └─ 调用 CBDESBEVFusion(**cfg.model)
      │
      └─ 将 cfg.model 字典的所有键值对作为关键字参数传递
          ├─ encoders=cfg.model['encoders']
          ├─ fuser=cfg.model['fuser']
          ├─ decoder=cfg.model['decoder']
          ├─ heads=cfg.model['heads']
          ├─ use_cbdes_moe=cfg.model['use_cbdes_moe']  ← 从配置文件读取
          └─ cbdes_moe_config=cfg.model['cbdes_moe_config']

  ↓
【步骤5】CBDESBEVFusion.__init__(...)
  │
  ├─ 接收参数: use_cbdes_moe=True (从配置文件)
  ├─ self.use_cbdes_moe = use_cbdes_moe
  │
  └─ 根据 use_cbdes_moe 决定是否使用 CBDESMoE
      ├─ if self.use_cbdes_moe:
      │   └─ 使用 cbdes_moe_config 创建 CBDESMoE
      │       └─ 加载各专家的预训练权重
      └─ else:
          └─ 使用配置文件中的 backbone (如 SwinTransformer)
```

## 配置文件继承关系

```
camera+lidar/default.yaml (基础配置)
  ↑ 继承
moe_lal/default.yaml (MoE 特定配置)
  ↑ 继承 (通过 _base_)
convfuser.yaml (ConvFuser 特定配置)
  └─ _base_: ['default.yaml']
```

## 配置合并示例

### 初始配置（camera+lidar/default.yaml）
```yaml
model:
  type: BEVFusion
  encoders:
    camera:
      backbone:
        type: SwinTransformer
```

### 继承配置（moe_lal/default.yaml）
```yaml
model:
  type: CBDESBEVFusion
  use_cbdes_moe: false
  cbdes_moe_config:
    expert_configs: {...}
```

### 最终配置（convfuser.yaml）
```yaml
_base_: ['default.yaml']

model:
  fuser:
    type: ConvFuser
```

### 合并后的最终结果
```yaml
model:
  type: CBDESBEVFusion          # 来自 moe_lal/default.yaml
  use_cbdes_moe: false          # 来自 moe_lal/default.yaml
  cbdes_moe_config: {...}       # 来自 moe_lal/default.yaml
  encoders:
    camera:
      backbone:
        type: SwinTransformer    # 来自 camera+lidar/default.yaml
  fuser:
    type: ConvFuser             # 来自 convfuser.yaml (覆盖)
```

## 关键代码位置

### 1. torchpack 配置加载
- **文件**: `torchpack/utils/config.py`
- **类**: `Config`
- **方法**: `load(fpath, recursive=True)`
- **功能**: 递归加载父目录的 default.yaml

### 2. mmcv 配置处理
- **文件**: `mmcv/utils/config.py` (在 anaconda 环境中)
- **类**: `Config`
- **方法**: `__init__(cfg_dict, filename=None)`
- **功能**: 处理 `_base_` 继承、变量替换、配置合并

### 3. 模型构建
- **文件**: `mmdet3d/models/builder.py`
- **函数**: `build_model(cfg)`
- **功能**: 将配置字典作为关键字参数传递给模型

### 4. 模型初始化
- **文件**: `mmdet3d/models/fusion_models/cbdes_bevfusion.py`
- **类**: `CBDESBEVFusion`
- **方法**: `__init__(..., use_cbdes_moe=True, ...)`
- **功能**: 根据 use_cbdes_moe 决定是否使用 CBDESMoE

## 调试技巧

### 1. 查看实际加载的配置
```python
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval

configs.load('configs/.../convfuser.yaml', recursive=True)
cfg = Config(recursive_eval(configs), filename='...')

# 查看完整配置
print(cfg.pretty_text)

# 查看特定值
print(cfg.model.use_cbdes_moe)
```

### 2. 追踪配置来源
```python
# 查看 torchpack 加载的配置
print(configs)

# 查看 mmcv 处理后的配置
print(cfg)
```

### 3. 验证配置合并
```python
# 运行演示脚本
python demo_config_loading.py
```

