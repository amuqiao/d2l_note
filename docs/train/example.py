#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""train.py 模块的测试用例

这个文件提供了train.py模块的各种使用场景的测试代码，包括：
- 基本训练流程
- 自定义训练参数
- 模型选择
- 可视化控制
- 模型保存控制

通过这些测试用例，您可以快速验证train.py模块的功能是否正常。"""

import os
import sys
import argparse
from src.trainer.trainer import Trainer
from src.utils.model_registry import ModelRegistry
from src.utils.log_utils import get_logger


def test_basic_training():
    """基本训练流程测试"""
    print("\n===== 基本训练流程测试 =====")
    
    # 初始化日志
    logger = get_logger(
        name="test",
        log_file="logs/test.log",
        global_level="DEBUG",
        console_level="INFO",
        file_level="DEBUG"
    )
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type="LeNet",
        num_epochs=3,
        lr=0.01,
        batch_size=128,
        input_size=(1, 1, 28, 28),
        resize=None,
        root_dir=os.getcwd()
    )
    
    # 执行训练
    try:
        result = trainer.run_training(
            model_type="LeNet",
            config=config,
            enable_visualization=False,
            save_every_epoch=False
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


def test_custom_parameters():
    """自定义训练参数测试"""
    print("\n===== 自定义训练参数测试 =====")
    
    # 初始化日志
    logger = get_logger(
        name="test",
        log_file="logs/test.log",
        global_level="DEBUG",
        console_level="INFO",
        file_level="DEBUG"
    )
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type="AlexNet",
        num_epochs=5,
        lr=0.001,
        batch_size=64,
        input_size=(1, 3, 224, 224),
        resize=224,
        root_dir=os.getcwd()
    )
    
    # 执行训练
    try:
        result = trainer.run_training(
            model_type="AlexNet",
            config=config,
            enable_visualization=False,
            save_every_epoch=True
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


def test_model_selection():
    """模型选择测试"""
    print("\n===== 模型选择测试 =====")
    
    # 初始化日志
    logger = get_logger(
        name="test",
        log_file="logs/test.log",
        global_level="DEBUG",
        console_level="INFO",
        file_level="DEBUG"
    )
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type="GoogLeNet",
        num_epochs=10,
        lr=0.001,
        batch_size=32,
        input_size=(1, 3, 224, 224),
        resize=224,
        root_dir=os.getcwd()
    )
    
    # 执行训练
    try:
        result = trainer.run_training(
            model_type="GoogLeNet",
            config=config,
            enable_visualization=False,
            save_every_epoch=False
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


def test_visualization_control():
    """可视化控制测试"""
    print("\n===== 可视化控制测试 =====")
    
    # 初始化日志
    logger = get_logger(
        name="test",
        log_file="logs/test.log",
        global_level="DEBUG",
        console_level="INFO",
        file_level="DEBUG"
    )
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type="ResNet",
        num_epochs=10,
        lr=0.001,
        batch_size=32,
        input_size=(1, 3, 224, 224),
        resize=224,
        root_dir=os.getcwd()
    )
    
    # 执行训练
    try:
        result = trainer.run_training(
            model_type="ResNet",
            config=config,
            enable_visualization=True,
            save_every_epoch=False
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


def test_save_control():
    """模型保存控制测试"""
    print("\n===== 模型保存控制测试 =====")
    
    # 初始化日志
    logger = get_logger(
        name="test",
        log_file="logs/test.log",
        global_level="DEBUG",
        console_level="INFO",
        file_level="DEBUG"
    )
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type="DenseNet",
        num_epochs=10,
        lr=0.001,
        batch_size=32,
        input_size=(1, 3, 96, 96),
        resize=96,
        root_dir=os.getcwd()
    )
    
    # 执行训练
    try:
        result = trainer.run_training(
            model_type="DenseNet",
            config=config,
            enable_visualization=False,
            save_every_epoch=True
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


def main():
    """主函数，运行所有测试用例"""
    print("train.py 模块测试用例")
    print("=" * 50)
    
    # 运行各个测试用例
    test_basic_training()
    test_custom_parameters()
    test_model_selection()
    test_visualization_control()
    test_save_control()
    
    print("\n" + "=" * 50)
    print("所有测试用例运行完毕。请查看控制台输出和生成的日志文件。")


if __name__ == "__main__":
    main()