#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试日志初始化时的打印信息功能"""
import os
import sys
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import reset_logger, Logger, init


def test_default_initialization():
    """测试默认初始化时的打印信息"""
    print("\n=== 测试默认初始化 ===")
    # 先重置日志系统
    reset_logger()
    # 默认初始化
    logger = Logger()
    print(f"日志是否已初始化: {Logger.is_initialized()}")
    print(f"日志文件路径: {logger.get_log_file_path()}")
    

def test_custom_initialization():
    """测试自定义初始化时的打印信息"""
    print("\n=== 测试自定义初始化 ===")
    # 先重置日志系统
    reset_logger()
    # 自定义初始化
    logger = init(
        log_dir="test_logs",
        log_file="test_init_print.log",
        log_level="DEBUG",
        console_level="INFO",
        use_timestamp=False
    )
    print(f"日志是否已初始化: {Logger.is_initialized()}")
    print(f"日志文件路径: {logger.get_log_file_path()}")
    

def test_direct_init_function():
    """测试直接调用init函数的打印信息"""
    print("\n=== 测试直接调用init函数 ===")
    # 先重置日志系统
    reset_logger()
    # 直接调用init函数
    from src.utils.logger import init as direct_init
    logger = direct_init(
        log_dir="test_logs",
        log_file="test_direct_init.log",
        log_level="INFO",
        console_level="WARNING",
        use_timestamp=True
    )
    print(f"日志是否已初始化: {Logger.is_initialized()}")
    print(f"日志文件路径: {logger.get_log_file_path()}")
    

def main():
    """运行所有测试"""
    print("开始测试日志初始化打印信息功能")
    test_default_initialization()
    test_custom_initialization()
    test_direct_init_function()
    print("\n所有测试完成！")
    

if __name__ == "__main__":
    main()