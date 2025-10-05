"""
model_center 包：提供通用的模型注册和训练框架

该包实现了一套完整的模型注册中心机制，支持命名空间，可以注册不同类型、自定义的模型，
并能够加载指定模型、数据加载器和参数配置来构建具体的训练过程。

核心组件：
- BaseRegistry: 通用组件注册中心基类
- ModelRegistry: 模型注册中心，用于管理和创建模型
- DataLoaderRegistry: 数据加载器注册中心，用于管理和创建数据集和数据加载器
- ConfigRegistry: 参数配置注册中心，用于管理和加载训练配置
- BaseTrainer: 训练器基类，提供训练和评估的基础功能
- ModelTrainerRegistry: 训练器注册中心

便捷函数：
- create_default_trainer: 创建默认训练器实例
- train_model: 训练模型的便捷函数

使用示例：
```python
from src.model_center import ModelRegistry, DataLoaderRegistry, ConfigRegistry, train_model

# 注册模型
@ModelRegistry.register(name="my_model", namespace="custom")
class MyModel(torch.nn.Module):
    pass

# 注册数据集
@DataLoaderRegistry.register_dataset(name="my_dataset", namespace="custom")
class MyDataset(torch.utils.data.Dataset):
    pass

# 加载配置
config = ConfigRegistry.get_config("my_config")

# 训练模型
trainer = train_model(config)
```
"""

# 导入核心注册中心类
from src.model_center.base_registry import ComponentRegistry
from src.model_center.model_registry import ModelRegistry
from src.model_center.data_loader_registry import DataLoaderRegistry
from src.model_center.config_registry import ConfigRegistry
from src.model_center.trainer import BaseTrainer, ModelTrainerRegistry

# 导入便捷函数
from src.model_center.trainer import create_default_trainer, train_model

# 导入日志配置
import logging

# 配置默认日志级别
logging.basicConfig(level=logging.INFO)

# 版本信息
__version__ = "1.0.0"

# 导出的公共API
__all__ = [
    # 注册中心类
    "ComponentRegistry",
    "ModelRegistry",
    "DataLoaderRegistry",
    "ConfigRegistry",
    "BaseTrainer",
    "ModelTrainerRegistry",
    
    # 便捷函数
    "create_default_trainer",
    "train_model",
    
    # 版本信息
    "__version__"
]


# 初始化函数
# 可以在这里添加全局初始化逻辑

def initialize():
    """初始化 model_center 包"""
    logging.info(f"model_center 包版本 {__version__} 已初始化")


# 自动初始化包
initialize()