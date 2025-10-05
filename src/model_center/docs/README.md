# model_center 模块文档

## 1. 模块介绍

model_center 是一个通用的模型注册和训练框架，提供了一套完整的组件注册中心机制，支持命名空间，可以注册不同类型、自定义的模型，并能够加载指定模型、数据加载器和参数配置来构建具体的训练过程。

**主要功能**：
- 统一的组件注册中心机制，支持模型、数据加载器和配置的注册与管理
- 基于命名空间的组件隔离，避免命名冲突
- 灵活的模型创建和配置加载机制
- 完整的训练流程支持，包括模型构建、优化器配置、训练和评估
- 便捷的装饰器语法，简化组件注册过程

## 2. 目录结构

model_center 模块的目录结构如下：

```
src/model_center/
├── __init__.py          # 包初始化文件，导出核心类和函数
├── base_registry.py     # 基础注册中心类定义
├── model_registry.py    # 模型注册中心实现
├── data_loader_registry.py # 数据加载器注册中心实现
├── config_registry.py   # 配置注册中心实现
├── trainer.py           # 训练器实现
├── examples/            # 示例代码
│   └── mnist_example.py # MNIST训练示例
└── docs/                # 文档目录
```

## 3. 核心组件

### 3.1 注册中心体系

model_center 采用层次化的注册中心设计，基于抽象基类定义统一接口，具体实现类提供特定领域的功能。

#### 3.1.1 BaseRegistry（抽象基类）

定义所有注册中心的统一接口，包含四个核心抽象方法：
- `register()`: 注册组件
- `unregister()`: 注销组件
- `get()`: 获取组件
- `list_components()`: 列出所有组件

#### 3.1.2 ComponentRegistry（通用实现）

提供注册中心的基本实现，管理组件的注册、注销、获取和列表功能。支持：
- 基于命名空间的组件隔离
- 装饰器模式的组件注册
- 自动处理组件名称

#### 3.1.3 ModelRegistry（模型注册中心）

专门用于管理深度学习模型的注册中心，扩展自ComponentRegistry，提供：
- 模型类注册和实例创建
- 模型配置管理
- 模型测试函数注册

#### 3.1.4 DataLoaderRegistry（数据加载器注册中心）

专门用于管理数据集和数据加载器的注册中心，扩展自ComponentRegistry，提供：
- 数据集类注册和实例创建
- 数据集配置管理
- 数据预处理函数注册

#### 3.1.5 ConfigRegistry（配置注册中心）

专门用于管理训练配置的注册中心，扩展自ComponentRegistry，提供：
- 配置模板注册
- 配置文件加载（支持JSON和YAML格式）
- 配置合并和验证

### 3.2 训练器

#### 3.2.1 BaseTrainer

训练器基类，提供训练和评估的基础功能，包括：
- 组件构建（模型、优化器、损失函数、数据加载器）
- 训练循环实现
- 模型评估
- 检查点保存和加载
- 早停机制

## 4. 工作流程

使用model_center进行模型训练的典型工作流程如下：

1. **注册组件**
   - 注册自定义模型到ModelRegistry
   - 注册数据集到DataLoaderRegistry
   - 注册配置模板或加载配置文件到ConfigRegistry

2. **创建训练配置**
   - 定义或加载包含模型、数据和训练参数的配置
   - 配置可以包含模型名称、数据集名称、优化器设置、训练参数等

3. **构建训练器**
   - 使用配置初始化BaseTrainer或其子类
   - 调用`build_components()`方法构建所需的组件

4. **执行训练**
   - 调用训练器的`train()`方法执行训练过程
   - 训练过程中自动保存检查点和最佳模型

5. **模型评估**
   - 训练完成后，可以使用训练器的`evaluate()`方法评估模型性能

## 5. 快速开始

### 5.1 基本使用示例

```python
from src.model_center import ModelRegistry, DataLoaderRegistry, ConfigRegistry, train_model

# 注册模型
@ModelRegistry.register(name="my_model", namespace="custom")
class MyModel(torch.nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 注册数据集
@DataLoaderRegistry.register_dataset(name="my_dataset", namespace="custom")
class MyDataset(torch.utils.data.Dataset):
    # 数据集实现...
    pass

# 创建配置
config = {
    "model": {
        "name": "my_model",
        "namespace": "custom",
        "args": {}
    },
    "data": {
        "train_dataset": {
            "name": "my_dataset",
            "namespace": "custom",
            "args": {}
        },
        "batch_size": 64
    },
    "training": {
        "epochs": 10,
        "optimizer": {
            "name": "adam",
            "args": {"lr": 0.001}
        }
    }
}

# 训练模型
trainer = train_model(config)
```

### 5.2 MNIST示例

model_center/examples/mnist_example.py 提供了一个完整的MNIST训练示例，展示了如何：
- 注册自定义CNN和MLP模型
- 注册MNIST数据集
- 定义和使用训练配置
- 执行模型训练和评估

## 6. 详细使用指南

### 6.1 模型注册与使用

#### 6.1.1 注册模型

使用装饰器注册模型：

```python
@ModelRegistry.register(name="lenet", namespace="cnn")
class LeNet(torch.nn.Module):
    # 模型实现...
    pass
```

或者使用函数式API注册模型：

```python
ModelRegistry.register(LeNet, name="lenet", namespace="cnn")
```

#### 6.1.2 注册模型配置

```python
# 注册模型时同时注册配置
@ModelRegistry.register_model(name="lenet", config={"input_channels": 1, "num_classes": 10})
class LeNet(torch.nn.Module):
    # 模型实现...
    pass

# 或者单独注册配置
ModelRegistry.register_config("lenet", {"input_channels": 1, "num_classes": 10})
```

#### 6.1.3 创建模型实例

```python
# 创建模型实例
model = ModelRegistry.create_model("lenet", namespace="cnn", input_channels=1, num_classes=10)
```

### 6.2 数据集注册与使用

#### 6.2.1 注册数据集

```python
@DataLoaderRegistry.register_dataset(name="mnist", namespace="datasets")
class MNISTDataset(torch.utils.data.Dataset):
    # 数据集实现...
    pass
```

#### 6.2.2 创建数据集和数据加载器

```python
# 创建数据集实例
train_dataset = DataLoaderRegistry.create_dataset("mnist", namespace="datasets", train=True)

test_dataset = DataLoaderRegistry.create_dataset("mnist", namespace="datasets", train=False)

# 创建数据加载器
train_loader = DataLoaderRegistry.create_data_loader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoaderRegistry.create_data_loader(test_dataset, batch_size=64)
```

### 6.3 配置管理

#### 6.3.1 注册配置模板

```python
# 注册配置模板
ConfigRegistry.register_template("mnist_config", {
    "model": {"name": "lenet", "namespace": "cnn"},
    "data": {"train_dataset": {"name": "mnist", "namespace": "datasets"}},
    "training": {"epochs": 10}
})
```

#### 6.3.2 从文件加载配置

```python
# 从JSON或YAML文件加载配置
config = ConfigRegistry.load_config_from_file("path/to/config.json")
```

### 6.4 使用训练器

#### 6.4.1 创建训练器

```python
# 使用配置创建训练器
trainer = BaseTrainer(config)

trainer.build_components()  # 构建模型、优化器、数据加载器等组件
```

#### 6.4.2 执行训练

```python
# 执行训练
trainer.train()

# 评估模型
trainer.evaluate()
```

#### 6.4.3 使用便捷函数

```python
# 使用便捷函数训练模型
trainer = train_model(config)
```

## 7. 最佳实践

### 7.1 组件注册

- **使用有意义的名称**：为注册的组件使用清晰、描述性的名称
- **合理使用命名空间**：使用命名空间对相关组件进行分组，避免命名冲突
- **提供默认配置**：为模型和数据集提供合理的默认配置，简化使用过程

### 7.2 配置管理

- **使用配置文件**：将复杂的配置存储在JSON或YAML文件中，方便修改和版本控制
- **分层配置**：按照模型、数据、训练等维度组织配置，提高可读性
- **配置验证**：在使用配置前进行验证，确保所有必需的参数都已提供

### 7.3 训练过程

- **设置适当的检查点**：定期保存模型检查点，避免训练中断导致的进度丢失
- **使用早停机制**：设置早停参数，防止过拟合
- **记录训练历史**：记录训练和验证指标，便于分析和可视化

## 8. 高级特性

### 8.1 命名空间管理

model_center的一个强大特性是支持命名空间，可以将组件分组存储，避免命名冲突。

```python
# 在不同命名空间注册同名组件
@ModelRegistry.register(name="model", namespace="experiment1")
class ModelV1(torch.nn.Module):
    pass

@ModelRegistry.register(name="model", namespace="experiment2")
class ModelV2(torch.nn.Module):
    pass

# 从不同命名空间获取组件
model1 = ModelRegistry.get("model", namespace="experiment1")
model2 = ModelRegistry.get("model", namespace="experiment2")
```

### 8.2 自定义训练器

可以通过继承BaseTrainer类创建自定义训练器，扩展或修改训练行为：

```python
class CustomTrainer(BaseTrainer):
    def __init__(self, config):
        super(CustomTrainer, self).__init__(config)
        # 添加自定义属性
        
    def _build_optimizer(self):
        # 自定义优化器构建逻辑
        
    def train_epoch(self, epoch):
        # 自定义训练轮次逻辑
```

## 9. 常见问题

### 9.1 组件未找到

如果出现"未找到组件"的错误，请检查：
- 组件名称是否正确
- 命名空间是否正确
- 组件是否已正确注册

### 9.2 配置错误

如果出现配置相关的错误，请检查：
- 配置格式是否正确
- 必需的配置项是否缺失
- 配置值的类型是否正确

### 9.3 训练失败

如果训练过程失败，请检查：
- 模型定义是否正确
- 数据格式是否与模型期望的输入格式匹配
- 设备配置是否正确（CPU/GPU）
- 内存使用是否过高

## 10. 示例代码

更多示例代码可以在 `model_center/examples/` 目录中找到，包括完整的MNIST训练示例。

## 11. 版本信息

当前版本：1.0.0

更新日志：
- 初始版本：实现基本的注册中心机制和训练功能
- 添加命名空间支持：增强组件管理能力
- 完善训练器功能：添加检查点、早停等功能

## 12. 联系方式

如有问题或建议，请联系：[项目维护者]