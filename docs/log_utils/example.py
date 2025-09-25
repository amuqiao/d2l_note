#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Log Utils 模块的使用示例

这个文件提供了log_utils模块的各种使用场景的示例代码，包括：
- 基本日志记录
- 自定义日志配置
- 日志级别设置
- 异常处理
- 多实例使用

通过这些示例，您可以快速了解如何在自己的项目中使用这个日志模块。
"""

import os
import sys
import logging
import time

# 添加项目根目录到Python搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 现在可以正确导入log_utils模块了
from src.utils.log_utils import get_logger, logger as default_logger


def basic_logging_example():
    """基本日志记录示例
    
    展示如何使用默认的logger实例进行基本的日志记录操作。
    """
    print("\n===== 基本日志记录示例 =====")
    
    # 使用默认的logger实例
    default_logger.debug("这是一条调试日志")
    default_logger.info("这是一条信息日志")
    default_logger.warning("这是一条警告日志")
    default_logger.error("这是一条错误日志")
    default_logger.critical("这是一条严重错误日志")


def custom_logger_example():
    """自定义日志配置示例
    
    展示如何创建自定义配置的日志实例，包括设置日志文件、日志级别等。
    """
    print("\n===== 自定义日志配置示例 =====")
    
    # 创建一个带文件输出的日志实例
    log_dir = "example_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"example_{int(time.time())}.log")
    
    custom_logger = get_logger(
        name="custom_logger",
        log_file=log_file,
        global_level=logging.DEBUG,
        console_level=logging.INFO,
        file_level=logging.DEBUG
    )
    
    print(f"日志文件已创建: {log_file}")
    custom_logger.debug("这条调试日志只会输出到文件")
    custom_logger.info("这条信息日志会同时输出到控制台和文件")
    custom_logger.warning("这条警告日志会同时输出到控制台和文件")
    custom_logger.error("这条错误日志会同时输出到控制台和文件")


def dynamic_level_change_example():
    """动态调整日志级别示例
    
    展示如何在运行时动态调整日志级别。
    """
    print("\n===== 动态调整日志级别示例 =====")
    
    dynamic_logger = get_logger(
        name="dynamic_logger",
        global_level=logging.INFO
    )
    
    # 初始级别设置为INFO
    print("初始级别: INFO")
    dynamic_logger.debug("这条调试日志不会显示")
    dynamic_logger.info("这条信息日志会显示")
    
    # 动态调整为DEBUG级别
    print("\n调整级别为: DEBUG")
    dynamic_logger.set_global_level(logging.DEBUG)
    dynamic_logger.debug("调整级别后，这条调试日志会显示")
    
    # 调整为ERROR级别
    print("\n调整级别为: ERROR")
    dynamic_logger.set_global_level(logging.ERROR)
    dynamic_logger.info("调整为ERROR级别后，这条信息日志不会显示")
    dynamic_logger.error("这条错误日志会显示")


def separate_levels_example():
    """独立设置控制台和文件日志级别示例
    
    展示如何为控制台和文件输出设置不同的日志级别。
    """
    print("\n===== 独立设置控制台和文件日志级别示例 =====")
    
    log_dir = "example_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"separate_levels_{int(time.time())}.log")
    
    separate_logger = get_logger(
        name="separate_logger",
        log_file=log_file,
        global_level=logging.DEBUG,
        console_level=logging.WARNING,  # 控制台只显示WARNING及以上级别
        file_level=logging.DEBUG        # 文件记录所有级别的日志
    )
    
    print(f"日志文件已创建: {log_file}")
    print("注意：控制台只显示WARNING及以上级别，而文件会记录所有级别的日志")
    
    separate_logger.debug("这条调试日志只会输出到文件")
    separate_logger.info("这条信息日志只会输出到文件")
    separate_logger.warning("这条警告日志会同时输出到控制台和文件")
    separate_logger.error("这条错误日志会同时输出到控制台和文件")


def exception_handling_example():
    """异常处理示例
    
    展示如何使用日志模块记录异常信息。
    """
    print("\n===== 异常处理示例 =====")
    
    exception_logger = get_logger(name="exception_logger")
    
    try:
        # 故意制造一个异常
        result = 1 / 0
    except Exception as e:
        print("捕获到异常，使用logger.exception记录详细信息：")
        exception_logger.exception("发生了除零异常")
        
    # 也可以选择不记录堆栈信息
    try:
        # 另一个异常
        values = [1, 2, 3]
        print(values[10])
    except Exception as e:
        print("\n捕获到异常，不记录堆栈信息：")
        exception_logger.exception("发生了索引越界异常", exc_info=False)


def simple_config_example():
    """简化配置示例
    
    展示如何使用简化的配置方式，只设置name=__name__和日志文件名。
    这是最常见的使用场景，适用于大多数简单应用。
    """
    print("\n===== 简化配置示例 =====")
    
    # 创建日志目录
    log_dir = "example_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"simple_config_{int(time.time())}.log")
    
    # 只设置name和日志文件名的简化配置
    simple_logger = get_logger(
        name=__name__,  # 使用当前模块名
        log_file=log_file  # 只指定日志文件路径
    )
    
    print(f"使用简化配置创建日志实例，日志文件：{log_file}")
    print("注意：这里只设置了name和log_file参数，其他参数使用默认值")
    
    simple_logger.info("这是使用简化配置记录的信息日志")
    simple_logger.warning("这是使用简化配置记录的警告日志")
    
    # 查看当前日志级别
    print(f"\n当前日志级别：{logging.getLevelName(simple_logger.logger.level)}")


def multiple_loggers_example():
    """多日志实例示例
    
    展示如何在同一个应用中使用多个不同的日志实例。
    """
    print("\n===== 多日志实例示例 =====")
    
    # 创建两个不同的日志实例
    module1_logger = get_logger(name="module1")
    module2_logger = get_logger(name="module2")
    
    module1_logger.info("这是来自模块1的日志")
    module2_logger.info("这是来自模块2的日志")
    
    # 可以为不同模块设置不同的日志级别
    module1_logger.set_global_level(logging.DEBUG)
    module2_logger.set_global_level(logging.WARNING)
    
    print("\n为不同模块设置不同的日志级别后：")
    module1_logger.debug("模块1的调试日志会显示")
    module2_logger.debug("模块2的调试日志不会显示")


def main():
    """主函数，运行所有示例"""
    print("Log Utils 模块使用示例")
    print("=" * 50)
    
    # 运行各个示例
    basic_logging_example()
    simple_config_example()  # 添加简化配置示例
    custom_logger_example()
    dynamic_level_change_example()
    separate_levels_example()
    exception_handling_example()
    multiple_loggers_example()
    
    print("\n" + "=" * 50)
    print("所有示例运行完毕。请查看控制台输出和生成的日志文件。")


if __name__ == "__main__":
    main()