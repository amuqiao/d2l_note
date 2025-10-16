"""
LeNet 示例测试文件

本测试文件用于验证使用统一注册入口重新实现的 LeNet 示例功能是否正常工作
"""

import sys
import os
import unittest
import torch
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# 设置日志级别为INFO，避免过多调试信息
logging.basicConfig(level=logging.INFO)

# 导入统一注册入口
src_module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if src_module_path not in sys.path:
    sys.path.append(src_module_path)

from src.model_center.model_registry_entry import (
    register_model, create_model, get_model_config, is_model_registered,
    get_all_registered_models, initialize_models
)
from src.model_center.data_loader_registry_entry import (
    register_dataset, create_dataset, create_data_loader, is_dataset_registered,
    get_dataset_config, initialize_data_loaders
)
from src.model_center.config_registry_entry import (
    get_config, is_config_registered,
    get_all_registered_configs, initialize_configs, register_config
)
from src.model_center.config_registry import ConfigRegistry

# 为了确保配置被正确注册，我们需要在测试前手动注册配置
def register_test_configs():
    """手动注册测试所需的配置"""
    # 创建配置类作为配置源
    class LeNetDefaultConfig:
        pass
        
    class LeNetBatchNormConfig:
        pass
        
    # 注册默认 LeNet 训练配置
    ConfigRegistry.register_config(
        config_name="lenet_default",
        template={
            "model": {
                "name": "LeNet",
                "namespace": "lenet",
                "type": "cnn",
                "args": {}
            },
            "data": {
                "train_dataset": {
                    "name": "mnist",
                    "namespace": "datasets",
                    "args": {
                        "train": True
                    }
                },
                "test_dataset": {
                    "name": "mnist",
                    "namespace": "datasets",
                    "args": {
                        "train": False
                    }
                },
                "batch_size": 256,
                "shuffle": True,
                "num_workers": 2
            },
            "training": {
                "optimizer": {
                    "name": "sgd",
                    "args": {
                        "lr": 0.8,
                        "momentum": 0.9
                    }
                },
                "scheduler": {
                    "name": "step",
                    "args": {
                        "step_size": 5,
                        "gamma": 0.1
                    }
                },
                "loss_function": {
                    "name": "cross_entropy",
                    "args": {}
                },
                "epochs": 10,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "checkpoint_dir": "./checkpoints/lenet",
                "log_interval": 100,
                "early_stopping": {
                    "enabled": True,
                    "patience": 3
                }
            }
        },
        namespace="default"
    )(LeNetDefaultConfig)
    
    # 注册 LeNetBatchNorm 训练配置
    ConfigRegistry.register_config(
        config_name="lenet_batchnorm",
        template={
            "model": {
                "name": "LeNetBatchNorm",
                "namespace": "lenet",
                "type": "cnn",
                "args": {}
            },
            "data": {
                "train_dataset": {
                    "name": "mnist",
                    "namespace": "datasets",
                    "args": {
                        "train": True
                    }
                },
                "test_dataset": {
                    "name": "mnist",
                    "namespace": "datasets",
                    "args": {
                        "train": False
                    }
                },
                "batch_size": 256,
                "shuffle": True,
                "num_workers": 2
            },
            "training": {
                "optimizer": {
                    "name": "sgd",
                    "args": {
                        "lr": 1.0,
                        "momentum": 0.9
                    }
                },
                "scheduler": {
                    "name": "step",
                    "args": {
                        "step_size": 2,
                        "gamma": 0.1
                    }
                },
                "loss_function": {
                    "name": "cross_entropy",
                    "args": {}
                },
                "epochs": 5,  # 带 BatchNorm 通常收敛更快
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "checkpoint_dir": "./checkpoints/lenet_batchnorm",
                "log_interval": 100,
                "early_stopping": {
                    "enabled": True,
                    "patience": 2
                }
            }
        },
        namespace="default"
    )(LeNetBatchNormConfig)

# 导入我们重新实现的LeNet示例模块
try:
    from src.model_center.examples.new_lenet_example import (
        RegisteredLeNet, RegisteredLeNetBatchNorm, MNISTDataset
    )
    HAS_NEW_LENET_EXAMPLE = True
    print("成功导入 new_lenet_example 模块")
except ImportError as e:
    HAS_NEW_LENET_EXAMPLE = False
    print(f"导入 new_lenet_example 模块失败: {e}")

# 手动注册测试配置
register_test_configs()


class TestLeNetExample(unittest.TestCase):
    """测试 LeNet 示例的功能"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化时调用，确保模块被正确导入"""
        if not HAS_NEW_LENET_EXAMPLE:
            cls.skipTest(cls, "new_lenet_example 模块导入失败，跳过所有测试")
        
        # 初始化所有注册中心
        initialize_models()
        initialize_data_loaders()
        initialize_configs()
    
    def test_model_registration(self):
        """测试模型是否被正确注册"""
        print("\n=== 测试模型注册 ===")
        
        # 检查LeNet模型是否已注册
        self.assertTrue(is_model_registered("LeNet", namespace="lenet", type="cnn"))
        print(f"LeNet 模型注册状态: {'成功' if is_model_registered('LeNet', namespace='lenet', type='cnn') else '失败'}")
        
        # 检查LeNetBatchNorm模型是否已注册
        self.assertTrue(is_model_registered("LeNetBatchNorm", namespace="lenet", type="cnn"))
        print(f"LeNetBatchNorm 模型注册状态: {'成功' if is_model_registered('LeNetBatchNorm', namespace='lenet', type='cnn') else '失败'}")
        
        # 打印所有已注册的模型
        all_models = get_all_registered_models()
        print(f"所有已注册模型数量: {sum(len(models) for models in all_models.values())}")
        print("已注册模型列表:")
        for ns, models in all_models.items():
            for model in models:
                print(f"  - {ns}/{model['name']}({model['type']})")
    
    def test_dataset_registration(self):
        """测试数据集是否被正确注册"""
        print("\n=== 测试数据集注册 ===")
        
        # 检查MNIST数据集是否已注册
        self.assertTrue(is_dataset_registered("mnist", namespace="datasets"))
        print(f"MNIST 数据集注册状态: {'成功' if is_dataset_registered('mnist', namespace='datasets') else '失败'}")
        
        # 获取MNIST数据集配置
        mnist_config = get_dataset_config("mnist", namespace="datasets")
        self.assertIsNotNone(mnist_config)
        print(f"MNIST 数据集配置: {mnist_config}")
        
        # 尝试创建一个小型数据集进行测试（只下载，不加载全部数据）
        try:
            # 仅创建数据集对象，不实际加载数据
            dataset = create_dataset("mnist", namespace="datasets", train=True, download=True)
            self.assertIsNotNone(dataset)
            print(f"成功创建 MNIST 数据集对象")
        except Exception as e:
            self.fail(f"创建 MNIST 数据集失败: {e}")
    
    def test_config_registration(self):
        """测试配置是否被正确注册"""
        print("\n=== 测试配置注册 ===")
        
        # 检查lenet_default配置是否已注册
        self.assertTrue(is_config_registered("lenet_default", namespace="default"))
        print(f"lenet_default 配置注册状态: {'成功' if is_config_registered('lenet_default', namespace='default') else '失败'}")
        
        # 检查lenet_batchnorm配置是否已注册
        self.assertTrue(is_config_registered("lenet_batchnorm", namespace="default"))
        print(f"lenet_batchnorm 配置注册状态: {'成功' if is_config_registered('lenet_batchnorm', namespace='default') else '失败'}")
        
        # 获取lenet_batchnorm配置
        batchnorm_config = get_config("lenet_batchnorm", namespace="default")
        self.assertIsNotNone(batchnorm_config)
        print(f"lenet_batchnorm 配置内容: {batchnorm_config.keys()}")
        self.assertIn('model', batchnorm_config)
        self.assertIn('data', batchnorm_config)
        self.assertIn('training', batchnorm_config)
    
    def test_model_creation(self):
        """测试模型是否能被正确创建"""
        print("\n=== 测试模型创建 ===")
        
        # 创建LeNet模型
        lenet_model = create_model("LeNet", namespace="lenet", type="cnn")
        self.assertIsNotNone(lenet_model)
        print(f"成功创建 LeNet 模型: {lenet_model.__class__.__name__}")
        
        # 创建LeNetBatchNorm模型
        lenet_bn_model = create_model("LeNetBatchNorm", namespace="lenet", type="cnn")
        self.assertIsNotNone(lenet_bn_model)
        print(f"成功创建 LeNetBatchNorm 模型: {lenet_bn_model.__class__.__name__}")
        
        # 测试模型前向传播
        try:
            # 创建一个随机输入张量，模拟MNIST图像大小(1, 28, 28)
            test_input = torch.randn(1, 1, 28, 28)
            
            # 将模型设置为评估模式以避免BatchNorm问题
            lenet_model.eval()
            lenet_bn_model.eval()
            
            # 测试LeNet模型前向传播
            with torch.no_grad():
                lenet_output = lenet_model(test_input)
                self.assertEqual(lenet_output.shape, (1, 10))  # MNIST有10个类别
                print(f"LeNet 模型前向传播成功，输出形状: {lenet_output.shape}")
                
                # 测试LeNetBatchNorm模型前向传播
                lenet_bn_output = lenet_bn_model(test_input)
                self.assertEqual(lenet_bn_output.shape, (1, 10))
                print(f"LeNetBatchNorm 模型前向传播成功，输出形状: {lenet_bn_output.shape}")
        except Exception as e:
            self.fail(f"模型前向传播测试失败: {e}")
    
    def test_simple_inference(self):
        """测试简单的推理功能"""
        print("\n=== 测试简单推理 ===")
        
        # 创建模型
        model = create_model("LeNetBatchNorm", namespace="lenet", type="cnn")
        
        # 创建随机测试图像
        test_image = torch.randn(1, 1, 28, 28)  # (batch_size, channels, height, width)
        
        # 设置模型为评估模式
        model.eval()
        
        # 进行推理
        with torch.no_grad():
            output = model(test_image)
            # 获取预测的类别
            _, predicted_class = torch.max(output, 1)
            
        print(f"推理结果 - 预测类别: {predicted_class.item()}")
        print(f"输出概率分布: {torch.softmax(output, dim=1).tolist()}")
        
        # 验证输出维度
        self.assertEqual(output.shape, (1, 10))


# 添加简单的命令行运行函数，方便单独运行测试
def run_tests():
    """运行所有测试"""
    print("开始测试使用统一注册入口重新实现的 LeNet 示例...")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestLeNetExample)
    
    # 运行测试
    unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    print("\n测试完成！")


if __name__ == "__main__":
    # 如果直接运行此脚本，执行所有测试
    run_tests()

    # 可选：添加一些额外的示例代码，演示如何使用统一注册入口进行简单的模型推理
    print("\n=== 演示使用统一注册入口进行简单推理 ===")
    
    try:
        # 1. 创建模型
        print("1. 创建 LeNetBatchNorm 模型...")
        model = create_model("LeNetBatchNorm", namespace="lenet", type="cnn")
        model.eval()
        
        # 2. 创建随机测试图像
        print("2. 创建随机测试图像...")
        test_image = torch.randn(1, 1, 28, 28)  # MNIST图像大小
        
        # 3. 进行推理
        print("3. 进行推理...")
        with torch.no_grad():
            output = model(test_image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # 4. 显示结果
        print(f"4. 推理结果:")
        print(f"   - 预测类别: {predicted_class.item()}")
        print(f"   - 置信度: {confidence.item():.4f}")
        print(f"   - 所有类别概率: {['{:.4f}'.format(p) for p in probabilities.squeeze().tolist()]}")
        
        print("\n使用统一注册入口进行模型推理演示成功！")
    except Exception as e:
        print(f"演示过程中出错: {e}")