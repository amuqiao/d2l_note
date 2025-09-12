# 日志模块使用指南

## 概述

LogUtils 是一个功能完整的日志管理工具类，专为 d2l_note 项目设计，提供统一的日志记录功能。该模块支持多种日志级别、控制台输出、文件记录、结构化日志等特性，帮助开发者更好地管理项目中的日志信息。

## 功能特性

- **多级别日志支持**：DEBUG、INFO、WARNING、ERROR、CRITICAL
- **双路输出**：同时输出到控制台和文件
- **结构化日志**：支持字典形式的结构化数据记录
- **异常处理**：提供异常日志记录功能，自动包含堆栈信息
- **单例模式**：全局唯一的日志实例，避免重复初始化
- **自定义配置**：支持自定义日志目录、文件名、级别等
- **友好的可视化**：使用图标增强日志的可读性

## 安装与使用

### 基本用法

日志模块已集成到项目中，您可以直接导入并使用：

```python
# 导入日志工具
from src.utils.log_utils import logger, debug, info, success, warning, error, critical

# 记录不同级别的日志
info("这是一条信息日志")
success("这是一条成功日志")
warning("这是一条警告日志")
error("这是一条错误日志")
```

### 初始化配置

LogUtils提供了两种初始化方式：默认初始化和显式初始化，以适应不同的使用场景。

### 默认初始化

LogUtils模块在导入时会自动创建一个全局logger实例，该实例默认配置了控制台输出，您可以直接使用而无需额外初始化：

```python
# 导入后直接使用（默认初始化）
from src.utils.log_utils import info

# 直接记录日志
info("这是一条使用默认初始化的日志")
```

默认初始化的特点：
- 自动配置控制台输出，显示INFO及以上级别的日志
- 自动创建带时间戳的日志文件（默认保存在项目根目录的logs文件夹）
- 使用默认的日志级别（INFO）

### 显式初始化 (init_logger)

对于需要自定义配置的场景，您可以使用`init_logger()`函数进行显式初始化：

```python
from src.utils.log_utils import init_logger

# 基本初始化（使用默认配置）
init_logger()

# 自定义配置
init_logger(
    log_dir="custom_logs",  # 自定义日志目录
    log_file="my_app.log",  # 自定义日志文件名
    log_level="DEBUG",      # 设置日志级别
    use_timestamp=True      # 文件名包含时间戳
)
```

显式初始化适用于以下场景：
- 需要自定义日志文件路径和名称
- 需要调整日志级别
- 需要控制是否在文件名中包含时间戳
- 希望在程序启动时明确初始化日志系统

## 多脚本调用场景下的初始化管理

在项目中，经常会出现多个脚本文件相互调用的情况，这可能导致日志系统被多次初始化。LogUtils通过单例模式解决了这个问题，但仍需要遵循一些最佳实践来确保日志系统正常工作。

### 多次初始化问题的解决方案

LogUtils模块采用单例模式设计，无论调用多少次`init_logger()`函数，全局都只会存在一个日志实例。但是，多次调用`init_logger()`仍可能导致以下问题：
- 日志配置被覆盖
- 日志文件被重新创建
- 重复添加日志处理器

为了避免这些问题，建议遵循以下原则：

1. **统一初始化**：在项目的主入口文件中进行一次初始化，其他模块直接使用即可
2. **避免在导入时初始化**：不要在模块的全局作用域中调用`init_logger()`，而是在函数内部（如`main()`函数）调用
3. **检查初始化状态**：在调用`init_logger()`前，可以检查日志实例是否已经初始化

### 推荐的项目集成模式

#### 主入口文件初始化

```python
# run.py 主入口文件
from src.utils.log_utils import init_logger, info

def main():
    # 在主函数中初始化日志系统
    init_logger(log_file="main_app.log")
    info("应用程序启动")
    
    # 调用其他模块
    from other_module import do_something
    do_something()

if __name__ == "__main__":
    main()
```

#### 被调用模块的使用方式

```python
# other_module.py 被调用的模块
from src.utils.log_utils import info

# 直接使用日志功能，无需再次初始化
def do_something():
    info("执行任务")
    # 业务逻辑
```

### 处理特殊情况

如果您需要在不同模块中使用不同的日志配置，可以考虑以下方法：

1. **使用日志名称区分**：为不同模块的日志使用不同的名称
2. **配置子日志器**：为特定模块配置子日志器
3. **使用配置文件**：通过统一的配置文件管理所有日志设置

### 日志级别说明

| 级别 | 描述 | 使用场景 |
|------|------|----------|
| DEBUG | 调试信息，最详细 | 开发调试阶段，记录详细的运行状态 |
| INFO | 一般信息 | 记录程序正常运行的关键信息 |
| WARNING | 警告信息 | 记录可能的问题，但不影响程序运行 |
| ERROR | 错误信息 | 记录错误情况，可能影响部分功能 |
| CRITICAL | 严重错误 | 记录致命错误，程序可能无法继续运行 |

### 详细功能示例

#### 1. 基本日志记录

```python
from src.utils.log_utils import debug, info, success, warning, error, critical

# 记录简单日志
info("项目启动成功")

# 使用格式化字符串
name = "模型训练" 
status = "已完成"
info("%s %s", name, status)

# 使用f-string
info(f"{name} {status}")
```

#### 2. 异常日志记录

```python
from src.utils.log_utils import exception, error

try:
    # 可能引发异常的代码
    result = 10 / 0
except Exception:
    # 自动记录异常详情和堆栈
    exception("计算过程中发生错误")
    
    # 或者手动记录异常
    error("捕获到异常: 除零错误")
```

#### 3. 结构化日志记录

```python
from src.utils.log_utils import log_dict

# 创建结构化数据
config = {
    "model_name": "LeNet",
    "num_epochs": 10,
    "learning_rate": 0.01,
    "batch_size": 128
}

# 记录结构化数据
log_dict("训练配置", config)

# 使用不同日志级别
log_dict("详细配置信息", config, log_level="DEBUG")
```

## 项目集成指南

### 在主程序中集成

建议在项目的主入口文件中初始化日志系统：

```python
# run.py 或其他主入口文件
from src.utils.log_utils import init_logger, info

def main():
    # 初始化日志系统
    init_logger(log_level="INFO")
    info("程序启动")
    
    # 程序的其他逻辑
    # ...

if __name__ == "__main__":
    main()
```

### 在工具类中使用

在项目的其他模块中，可以直接使用全局 logger：

```python
# 在其他模块中
from src.utils.log_utils import logger

class MyClass:
    def my_method(self):
        logger.info("执行我的方法")
        # ...
```

### 替换现有的 print 语句

为了统一日志管理，建议用日志模块替换项目中的 print 语句：

**替换前：**
```python
print("✅ 训练完成")
print(f"📊 最佳准确率: {best_acc:.4f}")
print(f"❌ 错误: {str(e)}")
```

**替换后：**
```python
from src.utils.log_utils import success, info, error

success("训练完成")
info(f"最佳准确率: {best_acc:.4f}")
error(f"错误: {str(e)}")
```

## 最佳实践

1. **统一初始化**：在程序入口处统一初始化日志系统
2. **合理使用日志级别**：根据信息重要程度选择合适的日志级别
3. **详细的错误日志**：记录异常时包含足够的上下文信息
4. **结构化记录配置和指标**：使用 log_dict 记录配置参数和性能指标
5. **日志文件管理**：定期清理或归档日志文件，避免占用过多磁盘空间
6. **避免日志膨胀**：在生产环境中适当提高日志级别，减少日志量

## 运行示例

您可以运行示例文件来了解日志模块的各项功能：

```bash
python src/utils/examples/log_example.py
```

示例程序会展示基本日志记录、文件日志、异常日志、结构化日志等功能，并生成日志文件供您查看。

## 注意事项

1. 日志模块采用单例模式，多次初始化不会创建新的实例
2. 默认情况下，控制台只显示 INFO 及以上级别的日志，而文件会记录所有级别的日志
3. 日志文件默认保存在项目根目录的 `logs` 文件夹中
4. 为了保持日志的一致性，建议在整个项目中统一使用该日志模块

## 版本更新

- v1.0：初始版本，提供基本的日志记录功能
- v1.1：增加结构化日志和异常处理功能
- v1.2：优化日志格式和图标显示

## 常见问题

**Q: 如何查看生成的日志文件？**
A: 默认情况下，日志文件保存在项目根目录的 `logs` 文件夹中，文件名包含日期时间戳。

**Q: 如何修改日志级别？**
A: 可以在初始化时通过 `log_level` 参数设置，例如 `init_logger(log_level="DEBUG")`。

**Q: 如何禁用控制台输出？**
A: 目前版本暂不支持直接禁用控制台输出，如需此功能，请联系开发人员。

**Q: 日志文件会自动滚动吗？**
A: 目前版本暂不支持日志文件自动滚动，建议定期手动清理或扩展该功能。

**Q: 多个脚本相互调用时，如何避免日志系统被多次初始化？**
A: LogUtils采用单例模式设计，全局只存在一个日志实例。为避免多次初始化导致的问题，建议：
- 在主入口文件中统一初始化一次
- 在模块中避免在全局作用域调用`init_logger()`
- 只在需要自定义配置时使用显式初始化

**Q: 默认初始化和显式调用`init_logger()`有什么区别？**
A: 默认初始化会在导入模块时自动创建全局logger实例，使用默认配置；而显式调用`init_logger()`允许您自定义日志配置，如日志文件路径、名称、级别等。如果您对日志配置没有特殊要求，可以直接使用默认初始化。