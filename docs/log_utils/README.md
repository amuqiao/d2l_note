# Log Utils 日志模块

## 功能介绍

这是一个功能完善的自定义日志模块，基于Python标准库的logging模块扩展实现。主要特性包括：

- 支持控制台和文件双重输出
- 日志级别带彩色图标标记，直观区分不同严重程度
- 可分别设置全局、控制台和文件的日志级别
- 自动创建日志目录，无需手动管理
- 异常处理功能，记录详细的错误堆栈
- 简洁易用的API接口

## 核心组件

### CustomLogger 类

自定义日志类，是整个模块的核心组件，提供了完整的日志功能。

### IconFormatter 内部类

自定义的日志格式化器，为不同级别的日志添加对应的图标标记。

### get_logger 函数

工厂函数，用于创建和配置日志实例，简化使用流程。

## 使用方法

### 基本用法

最简单的使用方式是直接导入默认的logger实例：

```python
from src.utils.log_utils import logger

logger.info("这是一条信息日志")
logger.warning("这是一条警告日志")
logger.error("这是一条错误日志")
```

### 创建自定义日志实例

通过`get_logger`函数创建自定义配置的日志实例：

```python
from src.utils.log_utils import get_logger
import logging

# 创建一个带文件输出的日志实例
logger = get_logger(
    name="my_app",
    log_file="logs/my_app.log",
    global_level=logging.DEBUG,
    console_level=logging.INFO,
    file_level=logging.DEBUG
)

logger.debug("这条调试日志只会输出到文件")
logger.info("这条信息日志会同时输出到控制台和文件")
```

### 设置不同的日志级别

可以在初始化时设置，也可以在运行时动态调整：

```python
# 初始化时设置
logger = get_logger(
    name="my_app",
    global_level=logging.INFO,
    console_level=logging.WARNING,
    file_level=logging.DEBUG
)

# 运行时调整
logger.set_global_level(logging.DEBUG)
logger.set_console_level(logging.ERROR)
logger.set_file_level(logging.INFO)
```

### 异常处理

使用exception方法记录异常信息：

```python
try:
    # 一些可能抛出异常的代码
    result = 1 / 0
except Exception as e:
    logger.exception("发生了异常")
    # 继续处理或抛出
```

## 日志图标说明

| 日志级别 | 图标 | 描述 |
|---------|------|------|
| DEBUG   | 🔍   | 调试信息，用于开发阶段排查问题 |
| INFO    | ✅   | 一般信息，记录正常的运行状态 |
| WARNING | ⚠️   | 警告信息，表示可能存在的问题但不影响程序运行 |
| ERROR   | ❌   | 错误信息，表示发生了错误但程序仍能继续运行 |
| CRITICAL | 🔥  | 严重错误，表示发生了致命错误，程序可能无法继续运行 |

## 日志格式

### 控制台输出格式

```
2023-11-01 12:34:56 ✅ INFO: 这是一条信息日志
```

### 文件输出格式

文件输出包含更详细的信息，包括模块名、函数名和行号：

```
2023-11-01 12:34:56 [✅] INFO module_name.function_name:123 - 这是一条信息日志
```

## 最佳实践

1. **为不同模块使用不同的日志名称**：有助于区分日志来源
   ```python
   logger = get_logger(name=__name__)
   ```

2. **合理设置日志级别**：
   - 开发环境：DEBUG
   - 测试环境：INFO
   - 生产环境：WARNING或ERROR

3. **结合文件和控制台输出**：控制台用于实时查看，文件用于持久化存储和后期分析

4. **使用异常记录功能**：在异常处理中使用logger.exception记录完整的堆栈信息

5. **定期清理日志文件**：避免日志文件过大

## 常见问题

### Q: 为什么我的日志没有输出到文件？
A: 请检查日志文件路径是否正确，以及文件日志级别是否设置正确。

### Q: 如何更改日志格式？
A: 当前版本的日志格式是固定的，如果需要自定义格式，需要修改源代码中的IconFormatter类。

### Q: 日志文件的编码是什么？
A: 默认使用UTF-8编码，确保支持中文等非ASCII字符。

## 示例代码

详见本目录下的`example.py`文件，包含了各种使用场景的示例代码。