# 模型注册中心使用指南

本项目实现了一个灵活的模型注册中心机制，使添加新的深度学习模型变得简单高效，无需修改大量代码。本文档将介绍如何使用这个机制。

## 设计理念

模型注册中心采用**工厂模式**和**注册表模式**，实现以下目标：

1. **解耦模型定义与使用**：模型定义和使用逻辑分离
2. **统一管理模型配置**：集中存储和管理模型的默认参数配置
3. **自动发现和注册**：支持自动注册新模型
4. **易于扩展**：添加新模型只需少量代码，无需修改核心逻辑

## 核心组件

### 1. ModelRegistry 类

`src/utils/model_registry.py` 文件中定义了 `ModelRegistry` 类，它是整个机制的核心，提供了以下主要功能：

- 模型类注册与管理
- 模型配置存储与访问
- 模型实例创建
- 模型测试函数注册与调用

## 添加新模型的方法

### 方法一：使用装饰器注册（推荐）

在模型类定义时，直接使用 `register_model` 装饰器进行注册：

```python
from src.utils.model_registry import ModelRegistry
from torch import nn

# 定义模型默认配置
MODEL_CONFIGS = {
    "input_size": (1, 1, 28, 28),
    "resize": None,
    "lr": 0.1,
    "batch_size": 256,
    "num_epochs": 10
}

@ModelRegistry.register_model("MyNewModel", MODEL_CONFIGS)
class MyNewModel(nn.Module):
    """示例：新模型定义"""
    def __init__(self):
        super(MyNewModel, self).__init__()
        # 模型定义代码...
        self.features = nn.Sequential(
            # 特征提取层...
        )
        self.classifier = nn.Sequential(
            # 分类器层...
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 方法二：运行时动态注册

对于已有的模型类，可以在运行时使用 `auto_register_model` 方法进行注册：

```python
from src.utils.model_registry import ModelRegistry
from src.models.some_existing_model import ExistingModel

# 定义或导入模型配置
MODEL_CONFIGS = {
    "input_size": (1, 1, 224, 224),
    "resize": 224,
    "lr": 0.05,
    "batch_size": 128,
    "num_epochs": 15
}

# 动态注册模型
ModelRegistry.auto_register_model(
    model_class=ExistingModel, 
    model_name="ExistingModel",  # 可选，默认使用类名
    config=MODEL_CONFIGS
)
```

## 在 run.py 中集成

通过之前的修改，`run.py` 已经完全支持使用模型注册中心：

1. 自动导入所有已注册的模型
2. 使用 `ModelRegistry.create_model()` 创建模型实例
3. 使用 `ModelRegistry.get_test_func()` 获取并调用测试函数
4. 自动获取并使用模型的默认配置
5. 命令行参数自动适应已注册的模型列表

## 新增模型完整流程

添加一个新模型的完整步骤如下：

1. 在 `src/models/` 目录下创建新的模型文件（例如 `my_new_model.py`）
2. 在文件中定义模型类并使用装饰器注册
3. 在 `run.py` 中添加模型文件的导入语句（如果还没有的话）
4. 无需修改其他任何代码，新模型将自动集成到系统中

## 自定义测试函数

如果需要为特定模型定制测试函数，可以使用 `register_test_func` 方法：

```python
from src.utils.model_registry import ModelRegistry

# 自定义测试函数
def custom_test_func(net, input_size):
    """自定义模型测试函数"""
    # 实现自定义测试逻辑
    
# 注册测试函数
ModelRegistry.register_test_func("MyNewModel", custom_test_func)
```

## 示例：添加DenseNet模型（已实现）

我们已经通过模型注册中心机制添加了DenseNet模型，示例如下：

在 `src/models/dense_net.py` 中：

```python
from torch import nn
from d2l import torch as d2l

# 模型组件定义...

class DenseNet(nn.Module):
    """DenseNet模型实现"""
    def __init__(self, growth_rate=32, num_convs_in_dense_blocks=[4, 4, 4, 4]):
        super(DenseNet, self).__init__()
        # 模型实现代码...

# 模型配置
DENSENET_CONFIGS = {
    "input_size": (1, 1, 96, 96),
    "resize": 96,
    "lr": 0.1,
    "batch_size": 256,
    "num_epochs": 10
}
```

在 `run.py` 中，通过简单的导入和注册即可：

```python
# 导入模型
from src.models.dense_net import DenseNet, DENSENET_CONFIGS

# 注册模型配置
ModelRegistry.register_config("DenseNet", DENSENET_CONFIGS)
```

## 优势总结

使用模型注册中心机制的主要优势：

1. **简化代码结构**：消除了大量的条件判断语句
2. **提高可扩展性**：添加新模型只需少量代码，无需修改核心逻辑
3. **统一管理**：集中管理模型类、配置和测试函数
4. **向后兼容**：保留了原有的使用方式，确保平滑过渡
5. **自动发现**：命令行参数自动适应已注册的模型列表

通过这种设计，项目变得更加模块化和易于维护，新增模型的工作量大大减少。