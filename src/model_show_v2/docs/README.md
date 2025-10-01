# Model Show V2 模块文档

## 1. 模块介绍

Model Show V2 是一个用于解析、结构化和可视化深度学习模型信息的模块化工具。该模块能够处理多种类型的模型相关文件（配置文件、指标文件、模型文件等），将其转换为统一的数据结构，并提供丰富的可视化功能，帮助用户直观地理解和比较不同模型的特性和性能。

### 主要功能：
- 自动解析多种类型的模型相关文件
- 统一的模型信息数据结构
- 丰富的可视化展示方式
- 支持单模型分析和多模型比较
- 可扩展的插件式设计

## 2. 目录结构

```
src/model_show_v2/
├── __init__.py             # 包初始化文件
├── data_access.py          # 数据访问层，负责文件读取
├── data_models.py          # 数据模型定义
├── parser_registry.py      # 解析器注册中心统一入口
├── visualizer_registry.py  # 可视化器注册中心统一入口
├── parsers/                # 解析器实现目录
│   ├── base_model_parsers.py       # 基础解析器抽象类
│   ├── model_config_parser.py      # 配置文件解析器
│   ├── model_metrics_parser.py     # 指标文件解析器
│   └── model_file_parser.py        # 模型文件解析器
├── visualizers/            # 可视化器实现目录
│   ├── base_model_visualizers.py   # 基础可视化器抽象类
│   ├── config_file_visualizer.py   # 配置文件可视化器
│   ├── model_metrics_visualizer.py # 指标可视化器
│   └── model_file_visualizer.py    # 模型文件可视化器
├── docs/                   # 文档目录
├── parser_example/         # 解析器使用示例
└── visualizer_example/     # 可视化器使用示例
```

## 3. 核心概念

### 3.1 数据模型

模块定义了两个核心数据模型：

1. **MetricData**：存储单个指标的详细信息
   - `name`：指标名称
   - `metric_type`：指标类型（scalar、curve、matrix、image）
   - `data`：指标的具体数据
   - `source_path`：指标来源文件路径
   - `timestamp`：指标生成时间戳
   - `description`：指标描述
   - `tags`：分类标签

2. **ModelInfoData**：整合模型的所有信息
   - `name`：模型名称
   - `path`：模型存储路径
   - `model_type`：模型类型
   - `timestamp`：模型生成时间戳
   - `params`：模型参数配置
   - `metric_list`：模型关联的指标列表
   - 辅助方法：`add_metric`, `get_metric_by_name`, `get_metrics_by_type`, `remove_metric`

### 3.2 解析器系统

解析器负责将不同类型的文件转换为标准化的ModelInfoData对象。系统包含：

1. **BaseModelInfoParser**：所有解析器的基类，定义了统一接口
   - `support()`：判断是否支持指定文件
   - `parse()`：解析文件为ModelInfoData对象

2. **ModelInfoParserRegistry**：解析器注册中心
   - 管理所有解析器的注册
   - 根据文件路径自动匹配最合适的解析器
   - 提供统一的解析入口

3. **具体解析器实现**：
   - `ConfigFileParser`：解析JSON格式的配置文件
   - `MetricsFileParser`：解析模型性能指标文件
   - `ModelFileParser`：解析PyTorch或ONNX模型文件

### 3.3 可视化器系统

可视化器负责将ModelInfoData对象转换为可视化结果。系统包含：

1. **BaseModelVisualizer**：所有可视化器的基类，定义了统一接口
   - `support()`：判断是否支持指定的ModelInfoData
   - `visualize()`：可视化单个模型信息
   - `compare()`：比较多个模型信息

2. **ModelVisualizerRegistry**：可视化器注册中心
   - 管理所有可视化器的注册
   - 根据ModelInfoData自动匹配最合适的可视化器
   - 提供统一的可视化入口

3. **具体可视化器实现**：
   - `ConfigFileVisualizer`：可视化配置文件信息
   - `MetricsVisualizer`：可视化模型性能指标
   - `ModelFileVisualizer`：可视化模型文件信息

## 4. 使用方法

### 4.1 基本使用流程

1. **导入必要的模块**

```python
from src.model_show_v2.parser_registry import parse_model_info
from src.model_show_v2.visualizer_registry import visualize_model_info
```

2. **解析模型文件**

```python
# 解析配置文件
model_info = parse_model_info("path/to/model_config.json")

# 解析指标文件
model_info = parse_model_info("path/to/metrics.json")

# 解析模型文件
model_info = parse_model_info("path/to/model.pth")
```

3. **可视化模型信息**

```python
# 可视化模型信息
visualization_result = visualize_model_info(model_info)

# 打印可视化结果
if visualization_result and visualization_result["success"]:
    print(visualization_result["text"])
```

### 4.2 模型比较

```python
from src.model_show_v2.parser_registry import parse_model_info
from src.model_show_v2.visualizer_registry import compare_model_infos

# 解析多个模型
model1_info = parse_model_info("path/to/model1.pth")
model2_info = parse_model_info("path/to/model2.pth")

# 比较模型
comparison_result = compare_model_infos([model1_info, model2_info])

# 打印比较结果
if comparison_result and comparison_result["success"]:
    print(comparison_result["text"])
```

### 4.3 使用命名空间

```python
# 使用特定命名空间的解析器
model_info = parse_model_info("path/to/model_config.json", namespace="config")

# 使用特定命名空间的可视化器
visualization_result = visualize_model_info(model_info, namespace="config")
```

## 5. 自定义扩展

Model Show V2 模块设计了灵活的插件系统，允许用户轻松添加自定义的解析器和可视化器。以下是详细的扩展步骤：

### 5.1 创建自定义解析器

#### 步骤1：实现解析器类

```python
from src.model_show_v2.parsers.base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData, MetricData
from src.utils.log_utils import get_logger

logger = get_logger(name="custom_parser")

@ModelInfoParserRegistry.register(namespace="custom")
class CustomFileParser(BaseModelInfoParser):
    # 设置优先级（可选，默认为50，数值越高优先级越高）
    priority = 60
    
    def support(self, file_path, namespace="default"):
        # 判断是否支持该文件
        # 这里示例支持.custom扩展名的文件
        return file_path.endswith(".custom")
    
    def parse(self, file_path, namespace="default"):
        # 解析文件并返回ModelInfoData对象
        try:
            # 示例：从.custom文件中提取信息
            # 实际应用中，根据文件格式实现相应的解析逻辑
            model_name = "custom_model"
            model_type = "custom_type"
            timestamp = os.path.getmtime(file_path)
            
            # 创建ModelInfoData对象
            model_info = ModelInfoData(
                name=model_name,
                path=file_path,
                model_type=model_type,
                timestamp=timestamp,
                framework="Custom Framework",
                task_type="custom_task",
                version="1.0",
                params={}
            )
            
            # 添加自定义指标
            custom_metric = MetricData(
                name="Custom Metric",
                metric_type="scalar",
                data={"value": 100, "unit": ""},
                source_path=file_path,
                timestamp=timestamp,
                description="A custom metric from .custom file"
            )
            model_info.add_metric(custom_metric)
            
            return model_info
        except Exception as e:
            logger.error(f"解析自定义文件失败 {file_path}: {str(e)}")
            raise ValueError(f"解析自定义文件失败: {str(e)}")
```

#### 步骤2：创建解析器测试用例

在`tests`目录下创建测试文件，例如`test_custom_parser.py`：

```python
import os
import tempfile
import unittest
from src.model_show_v2.parser_registry import parse_model_info

class TestCustomParser(unittest.TestCase):
    
    def setUp(self):
        # 创建临时测试文件
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.test_dir.name, "test_model.custom")
        
        # 写入测试内容（根据自定义解析器的预期格式）
        with open(self.test_file_path, "w") as f:
            f.write("# 这是一个测试用的自定义模型文件\n")
            f.write("model_name: test_model\n")
            f.write("custom_value: 100\n")
    
    def tearDown(self):
        # 清理临时文件
        self.test_dir.cleanup()
    
    def test_custom_parser(self):
        # 使用parse_model_info函数解析测试文件
        model_info = parse_model_info(self.test_file_path, namespace="custom")
        
        # 验证解析结果
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info.name, "custom_model")  # 根据解析器实现调整期望值
        self.assertEqual(model_info.model_type, "custom_type")
        self.assertEqual(len(model_info.metric_list), 1)
        self.assertEqual(model_info.metric_list[0].name, "Custom Metric")
    
    def test_parser_fallback(self):
        # 测试非支持文件格式的情况
        invalid_file = os.path.join(self.test_dir.name, "invalid.txt")
        with open(invalid_file, "w") as f:
            f.write("invalid content")
        
        model_info = parse_model_info(invalid_file, namespace="custom")
        self.assertIsNone(model_info)  # 应该返回None

if __name__ == "__main__":
    unittest.main()
```

#### 步骤3：注册解析器

解析器会通过装饰器`@ModelInfoParserRegistry.register`自动注册到系统中。确保在应用启动时导入了解析器模块，或者在`parser_registry.py`中添加导入：

```python
# 在parser_registry.py中添加
from .parsers.custom_file_parser import CustomFileParser
```

### 5.2 创建自定义可视化器

#### 步骤1：实现可视化器类

```python
import os
from typing import Dict, Any, List
from datetime import datetime
from prettytable import PrettyTable
from src.model_show_v2.visualizers.base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry
from src.model_show_v2.data_models import ModelInfoData
from src.utils.log_utils import get_logger

logger = get_logger(name="custom_visualizer")

@ModelVisualizerRegistry.register(namespace="custom")
class CustomVisualizer(BaseModelVisualizer):
    # 设置优先级（可选，默认为50）
    priority = 60
    
    def support(self, model_info, namespace="default"):
        # 判断是否支持该模型信息
        return model_info.model_type == "custom_type"
    
    def visualize(self, model_info, namespace="default"):
        # 实现可视化逻辑，使用prettytable创建美观表格
        try:
            # 创建主表格
            table = PrettyTable()
            table.title = f"自定义模型信息 ({model_info.name})"
            table.field_names = ["属性", "值"]
            
            # 添加基本信息
            table.add_row(["模型名称", model_info.name])
            table.add_row(["模型类型", model_info.model_type])
            table.add_row(["框架", model_info.framework])
            table.add_row(["任务类型", model_info.task_type])
            table.add_row(["版本", model_info.version])
            
            # 添加分割线
            table.add_row(["="*20, "="*40])
            
            # 添加自定义信息
            if model_info.params:
                table.add_row(["自定义参数", ""])
                for key, value in model_info.params.items():
                    table.add_row([f"  • {key}", str(value)])
            
            # 添加指标信息
            if model_info.metric_list:
                table.add_row(["="*20, "="*40])
                table.add_row(["指标信息", ""])
                for metric in model_info.metric_list:
                    value_str = f"{metric.data.get('value', '-')}{metric.data.get('unit', '')}"
                    table.add_row([f"  • {metric.name}", f"{value_str} ({metric.description})"])
            
            # 美化表格
            table.align["属性"] = "l"
            table.align["值"] = "l"
            
            return {
                "success": True,
                "message": "自定义模型可视化成功",
                "table": table,
                "text": str(table)
            }
        except Exception as e:
            logger.error(f"自定义模型可视化失败: {str(e)}")
            return {
                "success": False,
                "message": f"自定义模型可视化失败: {str(e)}"
            }
    
    def compare(self, model_infos, namespace="default"):
        # 实现比较逻辑，使用prettytable创建美观的比较表格
        try:
            if len(model_infos) < 2:
                return {
                    "success": False,
                    "message": "比较需要至少2个模型信息"
                }
                
            # 创建比较表格
            table = PrettyTable()
            
            # 设置表头
            headers = ["比较项"]
            for i, model_info in enumerate(model_infos, 1):
                headers.append(f"模型 {i}: {model_info.name}")
            
            table.field_names = headers
            
            # 添加基本信息比较
            basic_info = [
                ("模型名称", lambda info: info.name),
                ("模型类型", lambda info: info.model_type),
                ("框架", lambda info: info.framework),
                ("任务类型", lambda info: info.task_type),
                ("版本", lambda info: info.version)
            ]
            
            for label, getter in basic_info:
                row = [label]
                for model_info in model_infos:
                    row.append(getter(model_info))
                table.add_row(row)
            
            # 添加分割线
            divider_row = ["="*20] + ["="*30 for _ in range(len(model_infos))]
            table.add_row(divider_row)
            
            # 收集所有唯一的指标名称
            all_metric_names = set()
            for model_info in model_infos:
                for metric in model_info.metric_list:
                    all_metric_names.add(metric.name)
            
            # 添加指标比较
            if all_metric_names:
                for metric_name in sorted(all_metric_names):
                    row = [metric_name]
                    for model_info in model_infos:
                        metric_value = "- 无 -"
                        for metric in model_info.metric_list:
                            if metric.name == metric_name:
                                metric_value = f"{metric.data.get('value', '-')}{metric.data.get('unit', '')}"
                                break
                        row.append(metric_value)
                    table.add_row(row)
            
            # 美化表格
            for field in headers:
                table.align[field] = "l"
            
            return {
                "success": True,
                "message": "自定义模型比较成功",
                "table": table,
                "text": str(table)
            }
        except Exception as e:
            logger.error(f"自定义模型比较失败: {str(e)}")
            return {
                "success": False,
                "message": f"自定义模型比较失败: {str(e)}"
            }
```

#### 步骤2：创建可视化器测试用例

在`tests`目录下创建测试文件，例如`test_custom_visualizer.py`：

```python
import unittest
from src.model_show_v2.data_models import ModelInfoData, MetricData
from src.model_show_v2.visualizer_registry import visualize_model_info, compare_model_infos

class TestCustomVisualizer(unittest.TestCase):
    
    def setUp(self):
        # 创建测试用的ModelInfoData对象
        self.model_info1 = ModelInfoData(
            name="test_model1",
            path="/path/to/test_model1",
            model_type="custom_type",
            timestamp=1620000000,
            framework="Custom Framework",
            task_type="custom_task",
            version="1.0",
            params={"param1": "value1"}
        )
        
        # 添加测试指标
        metric1 = MetricData(
            name="Custom Metric",
            metric_type="scalar",
            data={"value": 100, "unit": ""},
            source_path="/path/to/test_model1",
            timestamp=1620000000,
            description="A test metric"
        )
        self.model_info1.add_metric(metric1)
        
        # 创建第二个模型信息用于比较
        self.model_info2 = ModelInfoData(
            name="test_model2",
            path="/path/to/test_model2",
            model_type="custom_type",
            timestamp=1620000001,
            framework="Custom Framework",
            task_type="custom_task",
            version="1.1",
            params={"param1": "value2"}
        )
        
        metric2 = MetricData(
            name="Custom Metric",
            metric_type="scalar",
            data={"value": 120, "unit": ""},
            source_path="/path/to/test_model2",
            timestamp=1620000001,
            description="A test metric"
        )
        self.model_info2.add_metric(metric2)
    
    def test_visualize(self):
        # 测试可视化单个模型
        result = visualize_model_info(self.model_info1, namespace="custom")
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertIn("table", result)
        self.assertIn("test_model1", result["text"])
        self.assertIn("Custom Metric", result["text"])
    
    def test_compare(self):
        # 测试比较多个模型
        result = compare_model_infos([self.model_info1, self.model_info2], namespace="custom")
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertIn("table", result)
        self.assertIn("test_model1", result["text"])
        self.assertIn("test_model2", result["text"])
        self.assertIn("Custom Metric", result["text"])
        
    def test_unsupported_model(self):
        # 测试不支持的模型类型
        unsupported_model = ModelInfoData(
            name="unsupported",
            path="/path/to/unsupported",
            model_type="unsupported_type",
            timestamp=0
        )
        
        result = visualize_model_info(unsupported_model, namespace="custom")
        # 应该返回None，因为没有找到支持的可视化器
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
```

#### 步骤3：注册可视化器

可视化器会通过装饰器`@ModelVisualizerRegistry.register`自动注册到系统中。确保在应用启动时导入了可视化器模块，或者在`visualizer_registry.py`中添加导入：

```python
# 在visualizer_registry.py中添加
from .visualizers.custom_file_visualizer import CustomVisualizer
```

### 5.3 使用prettytable创建美观表格

在可视化器实现中，我们可以使用`prettytable`库创建美观的表格。以下是一些常用的技巧：

```python
from prettytable import PrettyTable

# 创建表格
table = PrettyTable()

# 设置表格标题
table.title = "模型信息概览"

# 设置表头
table.field_names = ["属性", "值"]

# 添加数据行
table.add_row(["模型名称", model_info.name])

# 设置对齐方式
table.align["属性"] = "l"  # 左对齐

# 设置表格样式
# 设置边框样式
# table.set_style(PLAIN_COLUMNS)
# table.border = True
# table.header = True
# table.hrules = ALL

# 添加多行数据
for key, value in data.items():
    table.add_row([key, value])

# 输出表格字符串
print(str(table))
```

常用的表格样式设置：
- `table.align[字段名]`：设置特定字段的对齐方式（"l"左对齐，"c"居中，"r"右对齐）
- `table.border`：是否显示边框
- `table.header`：是否显示表头
- `table.hrules`：水平线显示方式（FRAME, HEADER, ALL, NONE）
- `table.vrules`：垂直线显示方式（FRAME, ALL, NONE）
- `table.padding_width`：单元格内边距
- `table.max_width[字段名]`：设置特定字段的最大宽度

通过这些设置，可以根据需求创建美观、易读的表格输出。

## 6. 最佳实践

### 6.1 选择合适的解析器
- 对于配置文件，使用`ConfigFileParser`（识别包含"config"的.json文件）
- 对于指标文件，使用`MetricsFileParser`（识别包含"metrics"的.json文件）
- 对于模型文件，使用`ModelFileParser`（识别.pth, .pt, .bin, .onnx文件）

### 6.2 提高解析和可视化效率
- 使用命名空间来限制搜索范围
- 为自定义解析器和可视化器设置合适的优先级
- 对于大型模型文件，考虑使用`weights_only=True`参数（PyTorch 2.0+）

### 6.3 错误处理
- 始终检查解析和可视化结果的`success`字段
- 处理可能的文件不存在、格式错误等异常情况
- 使用日志系统记录关键操作和错误信息

### 6.4 性能优化
- 对于大量模型的比较，考虑分批处理
- 重用已解析的ModelInfoData对象
- 避免重复加载大型模型文件

## 7. 常见问题

### 7.1 为什么我的文件无法被解析？
- 检查文件路径是否正确
- 确认文件格式是否受支持
- 查看日志以获取详细错误信息

### 7.2 如何提高解析准确性？
- 为特定文件类型创建自定义解析器
- 调整解析器的优先级
- 确保文件格式符合预期规范

### 7.3 可视化结果不符合预期怎么办？
- 检查模型信息数据是否完整
- 确认使用了正确的可视化器
- 考虑创建自定义可视化器

## 8. 示例

模块包含示例目录，展示如何使用解析器和可视化器：
- `parser_example/`：解析器使用示例
- `visualizer_example/`：可视化器使用示例

请参考这些示例了解更多具体用法。