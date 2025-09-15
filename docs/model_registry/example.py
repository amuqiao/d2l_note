#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""模型注册中心测试示例

这个文件提供了模型注册中心模块的各种使用场景的示例代码，包括：
- 模型注册
- 模型创建
- 模型配置获取
- 模型测试函数调用

通过这些示例，您可以快速了解如何在自己的项目中使用这个模型注册中心模块。
"""

import os
import sys
import logging
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.model_registry import ModelRegistry
from src.models.lenet import LeNet
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet
from src.models.dense_net import DenseNet
from src.models.mlp import MLP
from src.utils.network_utils import NetworkUtils


def register_models():
    """注册模型示例"""
    print("\n===== 注册模型示例 =====")
    
    # 使用装饰器自动注册的模型
    print("已注册模型:", ModelRegistry.list_models())


def create_model_example():
    """创建模型实例示例"""
    print("\n===== 创建模型实例示例 =====")
    
    # 创建LeNet模型实例
    lenet = ModelRegistry.create_model("LeNet")
    print(f"创建的模型类型: {type(lenet).__name__}")
    
    # 创建AlexNet模型实例
    alexnet = ModelRegistry.create_model("AlexNet")
    print(f"创建的模型类型: {type(alexnet).__name__}")


def get_config_example():
    """获取模型配置示例"""
    print("\n===== 获取模型配置示例 =====")
    
    # 获取LeNet模型配置
    lenet_config = ModelRegistry.get_config("LeNet")
    print(f"LeNet配置: {lenet_config}")
    
    # 获取AlexNet模型配置
    alexnet_config = ModelRegistry.get_config("AlexNet")
    print(f"AlexNet配置: {alexnet_config}")


def test_model_example():
    """测试模型示例"""
    print("\n===== 测试模型示例 =====")
    
    # 获取LeNet测试函数并执行
    lenet_test_func = ModelRegistry.get_test_func("LeNet")
    lenet_input_size = (1, 1, 28, 28)
    lenet_test_func(LeNet(), lenet_input_size)
    
    # 获取AlexNet测试函数并执行
    alexnet_test_func = ModelRegistry.get_test_func("AlexNet")
    alexnet_input_size = (1, 1, 224, 224)
    alexnet_test_func(AlexNet(), alexnet_input_size)


def main():
    """主函数，运行所有示例"""
    print("模型注册中心模块使用示例")
    print("=" * 50)
    
    # 运行各个示例
    register_models()
    create_model_example()
    get_config_example()
    test_model_example()
    
    print("\n" + "=" * 50)
    print("所有示例运行完毕。请查看控制台输出。")


if __name__ == "__main__":
    main()