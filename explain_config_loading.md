# 配置文件加载机制详解

## 1. 配置文件调用流程

### 1.1 训练脚本中的配置加载

```python
# tools/train_cbdes_moe.py 或 tools/train.py

# 步骤1: 使用 torchpack 加载配置文件
from torchpack.utils.config import configs
configs.load(args.config, recursive=True)  # 加载配置文件，recursive=True 会递归加载父目录的 default.yaml

# 步骤2: 处理命令行参数覆盖
configs.update(opts)  # opts 是命令行参数，如 --model.use_cbdes_moe=true

# 步骤3: 转换为 mmcv.Config 对象
from mmcv import Config
from mmdet3d.utils import recursive_eval
cfg = Config(recursive_eval(configs), filename=args.config)
# mmcv.Config 会自动处理 _base_ 继承
```

### 1.2 配置文件继承机制

#### torchpack.configs.load() 的 recursive 模式

当 `recursive=True` 时，torchpack 会：
1. 加载指定的配置文件（如 `convfuser.yaml`）
2. 递归向上查找父目录中的 `default.yaml` 文件
3. 按从根到叶的顺序合并配置（后加载的覆盖先加载的）

**示例路径查找顺序**：
```
configs/nuscenes/det/transfusion/secfpn/camera+lidar/moe_lal/convfuser.yaml
  ↓ 向上查找
configs/nuscenes/det/transfusion/secfpn/camera+lidar/moe_lal/default.yaml  ← 找到
configs/nuscenes/det/transfusion/secfpn/camera+lidar/default.yaml  ← 找到
configs/nuscenes/det/transfusion/secfpn/default.yaml  ← 可能找到
...
```

#### mmcv.Config 的 _base_ 继承

mmcv.Config 在初始化时会：
1. 读取配置文件中的 `_base_` 字段
2. 递归加载所有 `_base_` 中指定的配置文件
3. 按照加载顺序合并配置（后加载的覆盖先加载的）

**示例**：
```yaml
# convfuser.yaml
_base_: ['default.yaml']  # 继承 default.yaml

model:
  fuser:
    type: ConvFuser
    # 只覆盖 fuser 部分，其他配置从 default.yaml 继承
```

## 2. 配置合并顺序

### 2.1 完整的加载和合并流程

```
1. torchpack.configs.load(convfuser.yaml, recursive=True)
   ├─ 加载 convfuser.yaml
   ├─ 加载 moe_lal/default.yaml (父目录)
   ├─ 加载 camera+lidar/default.yaml (父目录)
   └─ 按顺序合并（后加载的覆盖先加载的）

2. mmcv.Config(recursive_eval(configs))
   ├─ 处理 _base_ 字段
   ├─ 加载 _base_ 中指定的所有配置文件
   └─ 合并配置（后加载的覆盖先加载的）

3. configs.update(opts)
   └─ 命令行参数覆盖（最高优先级）

4. 最终配置传递给模型
   └─ build_model(cfg.model) 将配置字典作为关键字参数传递
```

### 2.2 优先级顺序（从低到高）

1. **torchpack recursive 加载的父目录 default.yaml**（最低优先级）
2. **当前配置文件的内容**
3. **_base_ 中指定的配置文件**（按顺序，后加载的覆盖先加载的）
4. **命令行参数**（最高优先级）

## 3. 实际示例

### 3.1 配置文件结构

```
configs/
└── nuscenes/
    └── det/
        └── transfusion/
            └── secfpn/
                └── camera+lidar/
                    ├── default.yaml          # 基础配置
                    └── moe_lal/
                        ├── default.yaml      # MoE 特定配置
                        └── convfuser.yaml    # ConvFuser 特定配置
                            └── _base_: ['default.yaml']
```

### 3.2 加载 convfuser.yaml 的完整流程

```python
# 1. torchpack 递归加载
configs.load('convfuser.yaml', recursive=True)
# 实际加载顺序：
#   - camera+lidar/default.yaml
#   - moe_lal/default.yaml  
#   - convfuser.yaml

# 2. mmcv.Config 处理 _base_
cfg = Config(recursive_eval(configs), filename='convfuser.yaml')
# 处理 _base_: ['default.yaml']
#   - 加载 moe_lal/default.yaml（相对于 convfuser.yaml）
#   - 合并到当前配置

# 3. 命令行参数覆盖
configs.update(['--model.use_cbdes_moe=true'])
# 覆盖配置中的 use_cbdes_moe 值
```

## 4. 关键代码位置

### 4.1 torchpack 配置加载
- **文件**: `torchpack/utils/config.py`
- **方法**: `Config.load(fpath, recursive=True)`
- **功能**: 递归加载父目录的 default.yaml

### 4.2 mmcv 配置处理
- **文件**: `mmcv/config.py` (在 anaconda 环境中)
- **类**: `Config`
- **功能**: 处理 `_base_` 继承和配置合并

### 4.3 模型构建
- **文件**: `mmdet3d/models/builder.py`
- **函数**: `build_model(cfg)`
- **功能**: 将配置字典作为关键字参数传递给模型 `__init__`

## 5. 配置传递到模型的流程

```python
# 1. 配置文件 → torchpack.configs
configs.load('convfuser.yaml', recursive=True)
# configs 现在包含所有合并后的配置

# 2. torchpack.configs → mmcv.Config
cfg = Config(recursive_eval(configs), filename='convfuser.yaml')
# cfg.model 是一个字典，包含模型的所有配置

# 3. mmcv.Config → 模型实例
model = build_model(cfg.model)
# build_model 内部：
#   - 从 cfg.model 中提取 'type': 'CBDESBEVFusion'
#   - 调用 CBDESBEVFusion(**cfg.model)
#   - 将 cfg.model 中的所有键值对作为关键字参数传递
#   - 例如：use_cbdes_moe=cfg.model['use_cbdes_moe']
```

## 6. 注意事项

1. **配置覆盖顺序**: 后加载的配置会覆盖先加载的配置
2. **字典合并**: 字典类型的配置会递归合并，而不是完全替换
3. **命令行参数**: 使用 `--key=value` 或 `--key value` 格式
4. **嵌套配置**: 使用点号分隔，如 `--model.use_cbdes_moe=true`
5. **类型转换**: torchpack 会尝试使用 `literal_eval` 转换值类型

