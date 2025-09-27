# Model Show 模块文档

Model Show 模块是一个用于解析、可视化和比较深度学习模型信息的工具集，提供了灵活的插件式架构，支持不同类型的模型文件和配置格式。

## 目录结构

```
src/model_show/
├── __init__.py              # 模块初始化文件
├── data_access.py           # 数据访问层：负责数据的读取和解析
├── data_models.py           # 数据模型层：定义标准化的数据表示
├── parser_registry.py       # 解析器注册中心：管理所有解析器
├── visualizer_registry.py   # 可视化器注册中心：管理所有可视化器
├── parsers/                 # 解析器实现目录
│   ├── __init__.py
│   ├── base_model_parsers.py    # 解析器基类
│   ├── model_config_parser.py   # 配置文件解析器
│   ├── model_file_parser.py     # 模型文件解析器
│   └── model_metrics_parser.py  # 指标文件解析器
├── visualizers/             # 可视化器实现目录
│   ├── __init__.py
│   ├── base_model_visualizers.py    # 可视化器基类
│   └── config_file_visualizer.py    # 配置文件可视化器
├── parser_example/          # 解析器使用示例
├── visualizer_example/      # 可视化器使用示例
└── docs/                    # 文档目录
```

## 核心组件介绍

### 1. 数据模型层

数据模型层定义了统一的数据表示格式，确保不同来源的模型信息能够以标准化的方式处理。

#### ModelInfoData

统一表示模型和训练任务的信息，包含以下主要字段：

- **name**: 模型名称
- **path**: 模型存储路径
- **model_type**: 模型类型（如"CNN"、"Transformer"）
- **timestamp**: 创建/训练完成时间戳
- **params**: 模型参数配置
- **metrics**: 关键性能指标
- **framework**: 深度学习框架（如"PyTorch"、"TensorFlow"）
- **task_type**: 任务类型（如"classification"、"detection"）

#### MetricData

标准化表示各种指标数据，支持不同类型的指标（标量、曲线、矩阵、图像等）。

### 2. 数据访问层

DataAccessor类提供了统一的数据读取接口，支持多种文件格式：

```python
# 读取JSON文件
data = DataAccessor.read_file("config.json")

# 读取PyTorch模型文件
data = DataAccessor.read_file("model.pth")

# 获取文件时间戳
timestamp = DataAccessor.get_file_timestamp("config.json")
```

### 3. 解析器系统

解析器系统负责将不同格式的文件解析为标准化的ModelInfoData对象。

#### 基础解析器接口

BaseModelInfoParser定义了所有解析器必须实现的接口：

```python
class BaseModelInfoParser(ABC):
    priority: int = 50  # 解析器优先级
    
    @abstractmethod
    def support(self, file_path: str, namespace: str = "default") -> bool:
        # 判断是否支持该文件
        pass
        
    @abstractmethod
    def parse(self, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        # 解析文件为标准化ModelInfoData
        pass
```

#### 注册中心

ModelInfoParserRegistry管理所有解析器的注册和查找：

```python
# 注册解析器（装饰器用法）
@ModelInfoParserRegistry.register(namespace="config")
class ConfigFileParser(BaseModelInfoParser):
    # 解析器实现
    pass

# 自动匹配解析器并解析文件
model_info = ModelInfoParserRegistry.parse_file("config.json", namespace="config")
```

#### 内置解析器

- **ConfigFileParser**: 解析JSON格式的配置文件
- **ModelFileParser**: 解析模型文件（如PyTorch的.pth文件）
- **MetricsFileParser**: 解析指标文件

### 4. 可视化器系统

可视化器系统负责将ModelInfoData对象可视化为用户友好的格式，并支持多个模型信息的比较。

#### 基础可视化器接口

BaseModelVisualizer定义了所有可视化器必须实现的接口：

```python
class BaseModelVisualizer(ABC):
    priority: int = 50  # 可视化器优先级
    
    @abstractmethod
    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        # 判断是否支持该模型信息
        pass
        
    @abstractmethod
    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        # 将模型信息可视化为指定格式
        pass
        
    @abstractmethod
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        # 比较多个模型信息并可视化为指定格式
        pass
```

#### 注册中心

ModelVisualizerRegistry管理所有可视化器的注册和查找：

```python
# 注册可视化器（装饰器用法）
@ModelVisualizerRegistry.register(namespace="config")
class ConfigFileVisualizer(BaseModelVisualizer):
    # 可视化器实现
    pass

# 自动匹配可视化器并可视化模型
result = ModelVisualizerRegistry.visualize_model(model_info, namespace="config")

# 比较多个模型
comparison_result = ModelVisualizerRegistry.compare_models([model_info1, model_info2], namespace="config")
```

#### 内置可视化器

- **ConfigFileVisualizer**: 将配置文件可视化为表格格式，并支持多个配置文件的比较

## 使用指南

### 1. 解析模型信息

```python
from src.model_show.parser_registry import parse_model_info

# 解析配置文件
model_info = parse_model_info("path/to/config.json", namespace="config")

# 获取解析后的模型信息
print(f"模型名称: {model_info.name}")
print(f"模型类型: {model_info.model_type}")
print(f"模型参数: {model_info.params}")
```

### 2. 可视化模型信息

```python
from src.model_show.parser_registry import parse_model_info
from src.model_show.visualizers.config_file_visualizer import ConfigFileVisualizer

# 解析配置文件
model_info = parse_model_info("path/to/config.json", namespace="config")

# 创建可视化器并可视化
visualizer = ConfigFileVisualizer()
result = visualizer.visualize(model_info, namespace="config")

if result["success"]:
    print(result["text"])  # 打印表格
else:
    print(f"可视化失败: {result['message']}")
```

### 3. 比较多个模型信息

```python
from src.model_show.parser_registry import parse_model_info
from src.model_show.visualizers.config_file_visualizer import ConfigFileVisualizer

# 解析多个配置文件
model_info1 = parse_model_info("path/to/config1.json", namespace="config")
model_info2 = parse_model_info("path/to/config2.json", namespace="config")

# 创建可视化器并比较
visualizer = ConfigFileVisualizer()
result = visualizer.compare([model_info1, model_info2], namespace="config")

if result["success"]:
    print(result["text"])  # 打印比较表格
else:
    print(f"比较失败: {result['message']}")
```

### 4. 使用注册中心自动匹配

```python
from src.model_show.parser_registry import parse_model_info
from src.model_show.visualizer_registry import ModelVisualizerRegistry

# 解析配置文件
model_info = parse_model_info("path/to/config.json", namespace="config")

# 自动匹配可视化器并可视化
visualization_result = ModelVisualizerRegistry.visualize_model(model_info, namespace="config")
if visualization_result:
    print(visualization_result["text"])

# 解析多个配置文件并自动比较
model_info1 = parse_model_info("path/to/config1.json", namespace="config")
model_info2 = parse_model_info("path/to/config2.json", namespace="config")
comparison_result = ModelVisualizerRegistry.compare_models([model_info1, model_info2], namespace="config")
if comparison_result:
    print(comparison_result["text"])
```

## 扩展指南

### 1. 自定义解析器

要创建自定义解析器，需要继承BaseModelInfoParser并实现其接口，然后使用注册器注册：

```python
from src.model_show.parsers.base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show.data_models import ModelInfoData

@ModelInfoParserRegistry.register(namespace="custom")
class CustomFileParser(BaseModelInfoParser):
    # 设置优先级（数值越高优先级越高）
    priority = 60
    
    def support(self, file_path: str, namespace: str = "default") -> bool:
        # 判断是否支持该文件
        return file_path.endswith(".custom")
        
    def parse(self, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        # 实现解析逻辑
        # ...
        return ModelInfoData(
            name="custom_model",
            path=file_path,
            model_type="custom_type",
            timestamp=...,  # 时间戳
            params={...},   # 解析后的参数
            # 其他字段
        )
```

### 2. 自定义可视化器

要创建自定义可视化器，需要继承BaseModelVisualizer并实现其接口，然后使用注册器注册：

```python
from src.model_show.visualizers.base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry
from src.model_show.data_models import ModelInfoData

@ModelVisualizerRegistry.register(namespace="custom")
class CustomModelVisualizer(BaseModelVisualizer):
    # 设置优先级
    priority = 60
    
    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        # 判断是否支持该模型信息
        return model_info.namespace == "custom"
        
    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        # 实现可视化逻辑
        # ...
        return {
            "success": True,
            "message": "可视化成功",
            "text": "自定义可视化结果",
            # 其他可视化数据
        }
        
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        # 实现比较逻辑
        # ...
        return {
            "success": True,
            "message": "比较成功",
            "text": "自定义比较结果",
            # 其他比较数据
        }
```

## 最佳实践

1. **使用命名空间隔离解析器/可视化器**：为不同类型的解析器和可视化器使用不同的命名空间，避免命名冲突

2. **合理设置优先级**：通过设置合适的优先级，确保在多个解析器/可视化器都能处理同一文件时，选择最合适的那个

3. **错误处理**：在解析和可视化过程中，适当处理可能出现的错误，并提供清晰的错误信息

4. **数据格式化**：对于复杂的数据结构，进行适当的格式化处理，使输出结果更加友好和可读

5. **复用现有组件**：尽可能复用现有的数据模型、数据访问和解析器/可视化器组件，避免重复造轮子

## 示例代码

模块包含了丰富的示例代码，帮助你快速上手：

- **parser_example/**: 包含解析器的使用示例和测试代码
- **visualizer_example/**: 包含可视化器的使用示例和测试代码

你可以通过运行以下命令来测试配置文件的可视化功能：

```bash
python src/model_show/visualizer_example/direct_test_configs.py
```

## 日志配置

模块使用了统一的日志系统，日志文件存储在项目根目录的`logs/`文件夹下：

- `logs/model_info_parser.log`: 解析器日志
- `logs/model_visualizer.log`: 可视化器日志
- `logs/data_access.log`: 数据访问层日志

日志级别默认为INFO，可以通过修改代码中的`global_level`参数来调整日志级别。

## 注意事项

1. 确保项目根目录已添加到Python路径中，以便正确导入模块

2. 目前支持的文件格式包括：
   - 配置文件：JSON格式
   - 模型文件：PyTorch的.pth格式

3. 可视化结果以表格形式输出，使用PrettyTable库进行格式化

4. 扩展模块时，应遵循现有的接口定义和命名规范，确保与现有系统的兼容性