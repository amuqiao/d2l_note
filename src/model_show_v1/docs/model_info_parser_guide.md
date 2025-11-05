# 模型信息解析器框架 (Model Info Parser Framework)

## 功能介绍

Model Show 是一个灵活、可扩展的模型信息解析器框架，旨在提供统一的接口来解析不同格式的模型相关文件，提取标准化的模型信息。

该模块的主要功能包括：
- 集中管理和注册多种文件解析器
- 支持自动匹配最合适的解析器处理不同类型的文件
- 提供统一的接口来访问解析后的模型信息
- 支持解析器的命名空间隔离
- 提供便捷的函数简化外部调用

## 目录结构

```
src/model_show/
├── __init__.py          # 包初始化文件
├── data_access.py       # 数据访问工具
├── data_models.py       # 数据模型定义
├── parser_registry.py   # 解析器注册中心入口
├── parsers/             # 解析器实现
│   ├── __init__.py
│   ├── base_parsers.py  # 解析器基类和注册中心
│   ├── config_parser.py # 配置文件解析器
│   ├── metrics_parser.py # 指标文件解析器
│   └── model_parser.py  # 模型文件解析器
└── parser_example/      # 使用示例
    ├── example_usage.py  # 基础使用示例
    ├── test_config_parser.py  # 配置解析器测试
    ├── test_metrics_parser.py # 指标解析器测试
    └── test_model_parser.py   # 模型解析器测试
```

## 核心组件

### 1. 数据模型

`data_models.py` 定义了标准化的数据结构，用于存储解析后的模型信息：

- `ModelInfoData`: 存储模型的基本信息、参数、性能指标等

### 2. 解析器基类

`parsers/base_parsers.py` 定义了解析器的抽象接口和注册中心：

- `BaseModelInfoParser`: 所有解析器必须实现的抽象基类
- `ModelInfoParserRegistry`: 解析器注册中心，负责管理和匹配解析器

### 3. 具体解析器实现

模块提供了多种预定义的解析器：

- `ConfigFileParser`: 解析模型配置文件
- `MetricsFileParser`: 解析模型性能指标文件
- `ModelFileParser`: 解析模型文件（如.pth文件）

### 4. 解析器注册中心

`parser_registry.py` 是整个模块的入口，提供了以下功能：
- 集中导入所有解析器，触发自动注册
- 对外暴露核心类和便捷函数
- 提供解析器初始化和状态检查

## 使用方法

### 1. 基本使用

最简单的使用方式是通过 `parse_model_info` 便捷函数：

```python
from src.model_show.parser_registry import parse_model_info

# 解析文件\model_info = parse_model_info(
    file_path="path/to/your/file.json",
    namespace="default"
)

if model_info:
    print(f"模型名称: {model_info.name}")
    print(f"模型类型: {model_info.model_type}")
    print(f"参数信息: {model_info.params}")
    print(f"性能指标: {model_info.metrics}")
```

### 2. 直接使用注册中心

也可以直接使用 `ModelInfoParserRegistry` 类：

```python
from src.model_show.parser_registry import ModelInfoParserRegistry

# 解析文件
model_info = ModelInfoParserRegistry.parse_file(
    file_path="path/to/your/model.pth",
    namespace="models"
)
```

### 3. 查看已注册的解析器

`get_all_registered_parsers` 方法主要用于获取当前已注册的所有解析器信息，适用于以下场景：

- **调试和故障排查**：当解析器无法正常工作时，检查哪些解析器已被注册
- **动态发现**：应用程序需要动态了解可用的解析器类型
- **自定义扩展**：开发人员需要确认自己的自定义解析器是否正确注册
- **测试验证**：在单元测试中验证解析器注册机制是否正常工作

```python
from src.model_show.parser_registry import get_all_registered_parsers

# 获取所有已注册的解析器
all_parsers = get_all_registered_parsers()
print(f"已注册的解析器数量: {len(all_parsers)}")
for parser in all_parsers:
    print(f"- 解析器: {parser.__class__.__name__}, 优先级: {parser.priority}, 命名空间: {parser.namespace}")

# 获取特定命名空间的解析器
config_parsers = get_all_registered_parsers(namespace="config")
print(f"配置解析器数量: {len(config_parsers)}")
```

### 4. 初始化解析器系统

`initialize_parsers` 方法用于验证解析器注册状态并记录调试信息，主要适用于以下场景：

- **应用程序启动时**：在应用程序初始化阶段调用，确保所有解析器正确加载
- **开发环境**：开发过程中快速验证解析器注册是否成功
- **日志记录**：记录解析器的注册状态，便于后续问题排查
- **大规模应用**：在包含多个模块和解析器的复杂应用中，统一管理解析器初始化

**注意**：`example_usage.py` 等使用示例文件不需要显式调用此方法，因为解析器会在导入时自动注册，而 `parse_model_info` 函数内部会处理解析器的匹配和调用。

```python
from src.model_show.parser_registry import initialize_parsers

# 初始化解析器系统（验证注册是否成功并记录日志）
initialize_parsers()

# 可以指定日志级别，获取更详细的初始化信息
initialize_parsers(verbose=True)
```

## 示例代码

查看 `parser_example/example_usage.py` 了解更多使用示例：

```python
from src.model_show.parser_registry import parse_model_info
import os

# 解析指标文件\metrics_file = "runs/run_20250914_044631/metrics.json"
if os.path.exists(metrics_file):
    metrics_info = parse_model_info(
        file_path=metrics_file,
        namespace="metrics"
    )
    if metrics_info:
        print(f"解析指标文件: {metrics_info.name}")
        print(f"最终测试准确率: {metrics_info.metrics.get('final_test_acc', 0.0):.4f}")
        print(f"最佳测试准确率: {metrics_info.metrics.get('best_test_acc', 0.0):.4f}")
        print(f"最终训练损失: {metrics_info.metrics.get('final_train_loss', 0.0):.6f}")
```

## 创建自定义解析器

要创建自定义解析器，只需继承 `BaseModelInfoParser` 并实现必要的方法，然后使用装饰器注册：

```python
from src.model_show.parsers.base_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show.data_models import ModelInfoData

@ModelInfoParserRegistry.register(namespace="custom")
class CustomFileParser(BaseModelInfoParser):
    """自定义文件解析器"""
    
    # 设置解析器优先级（可选）
    priority: int = 60
    
    def support(self, file_path: str) -> bool:
        """判断是否支持该文件"""
        return file_path.endswith(".custom")
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        """解析文件并返回ModelInfoData对象"""
        # 实现解析逻辑
        # ...
        return ModelInfoData(
            name="自定义模型",
            path=file_path,
            model_type="custom",
            timestamp=...,  # 设置时间戳
            params={},      # 设置模型参数
            metrics={},     # 设置性能指标
            namespace="custom"
        )
```

## 最佳实践

1. **使用便捷函数**：优先使用 `parse_model_info` 函数简化调用

2. **指定命名空间**：为不同类型的文件使用合适的命名空间，提高解析效率

3. **错误处理**：始终检查解析结果是否为 None，处理文件不存在或格式不支持的情况

4. **初始化验证**：在应用启动时调用 `initialize_parsers()` 确保所有解析器正确注册

5. **自定义解析器**：创建新的解析器时遵循抽象接口，使用装饰器注册

6. **优先级设置**：为解析器设置合适的优先级，确保正确的匹配顺序

## 测试

模块包含了多个测试文件，用于验证解析器的功能：

```bash
# 运行配置文件解析器测试
python -m unittest src.model_show.parser_example.test_config_parser

# 运行指标文件解析器测试
python -m unittest src.model_show.parser_example.test_metrics_parser

# 运行模型文件解析器测试
python -m unittest src.model_show.parser_example.test_model_parser
```

## 日志

模块使用 `src.utils.log_utils` 记录运行日志，可通过以下方式配置日志级别：

```python
from src.utils.log_utils import get_logger

# 获取解析器模块的日志器并设置级别
logger = get_logger(name="parser_registry")
logger.set_global_level("INFO")  # 或 "DEBUG", "WARNING", "ERROR"
```

## 版本信息

- 版本：v1.0
- 最后更新：2024-09-15

## 扩展指南

如需扩展模块功能，可以考虑以下方向：

1. 添加新类型的文件解析器
2. 扩展 `ModelInfoData` 数据模型以支持更多信息
3. 增加解析器的配置选项
4. 添加缓存机制提高解析性能
5. 实现异步解析功能