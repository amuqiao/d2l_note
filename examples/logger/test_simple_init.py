#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试日志初始化信息是否已简化为一行提示"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import init, Logger, reset_logger


def test_default_initialization():
    """测试默认初始化场景"""
    print("\n=== 测试默认初始化 ===")
    reset_logger()  # 重置日志系统
    logger = Logger()  # 默认初始化
    print(f"初始化状态: {Logger.is_initialized()}")


def test_custom_initialization():
    """测试自定义初始化场景"""
    print("\n=== 测试自定义初始化 ===")
    reset_logger()  # 重置日志系统
    logger = init(log_dir="custom_logs", log_level="DEBUG", console_level="INFO")
    print(f"初始化状态: {Logger.is_initialized()}")


def test_direct_init_function():
    """测试直接调用init函数场景"""
    print("\n=== 测试直接调用init函数 ===")
    reset_logger()  # 重置日志系统
    logger = init(log_file="test.log", use_timestamp=False)
    print(f"初始化状态: {Logger.is_initialized()}")


if __name__ == "__main__":
    print("开始测试日志初始化信息简化功能...")
    test_default_initialization()
    test_custom_initialization()
    test_direct_init_function()
    print("\n所有测试完成！")