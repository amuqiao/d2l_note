"""
LeNet 训练示例：展示如何使用 model_center 包进行 LeNet 模型注册和训练

本示例演示了：
1. 如何导入并注册现有的 LeNet 模型
2. 如何注册数据集
3. 如何创建和加载训练配置
4. 如何使用训练器进行模型训练
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.model_center import ModelRegistry, DataLoaderRegistry, ConfigRegistry, train_model
from src.models.lenet import LeNet, LeNetBatchNorm


# ===================== 注册 LeNet 模型 =====================

# 从 src/models/lenet.py 导入 LeNet 和 LeNetBatchNorm 模型
# 并使用 model_center 的 ModelRegistry 重新注册它们
@ModelRegistry.register_model(model_name="LeNet", namespace="lenet", type="cnn", config={
    "num_classes": 10,
    "lr": 0.8,  # LeNet 适合稍高学习率
    "batch_size": 256
})
class RegisteredLeNet(LeNet):
    """注册到 model_center 的 LeNet 模型"""
    pass


@ModelRegistry.register_model(model_name="LeNetBatchNorm", namespace="lenet", type="cnn", config={
    "num_classes": 10,
    "lr": 1.0,  # 带 BatchNorm 的 LeNet 学习率
    "batch_size": 256
})
class RegisteredLeNetBatchNorm(LeNetBatchNorm):
    """注册到 model_center 的带 Batch Normalization 的 LeNet 模型"""
    pass


# ===================== 注册数据集 =====================

# 数据预处理转换
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
])


@DataLoaderRegistry.register_dataset(dataset_name="mnist", namespace="datasets", config={
    "root": "./data",
    "download": True,
    "transform": mnist_transform
})
class MNISTDataset:
    """MNIST 数据集包装类"""
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform=None):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# ===================== 注册配置模板 =====================

# 注册默认 LeNet 训练配置
ConfigRegistry.register_template("lenet_default", {
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
})


# 注册 LeNetBatchNorm 训练配置
ConfigRegistry.register_template("lenet_batchnorm", {
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
})


# ===================== 主函数 =====================

def main():
    """主函数：加载配置并训练模型"""
    # 选择要使用的配置
    # 可以使用 LeNet 的默认配置
    # config = ConfigRegistry.get_config("lenet_default")
    
    # 也可以使用带 BatchNorm 的配置
    config = ConfigRegistry.get_config("lenet_batchnorm")
    
    # 或者动态修改配置
    # config['training']['epochs'] = 15
    # config['training']['optimizer']['args']['lr'] = 0.01
    
    # 打印配置信息
    print("\n训练配置:")
    print(f"- 模型: {config['model']['name']}")
    print(f"- 设备: {config['training']['device']}")
    print(f"- 批次大小: {config['data']['batch_size']}")
    print(f"- 训练轮数: {config['training']['epochs']}")
    print(f"- 优化器: {config['training']['optimizer']['name']}")
    print(f"- 学习率: {config['training']['optimizer']['args']['lr']}")
    print(f"- 检查点目录: {config['training']['checkpoint_dir']}")
    
    # 训练模型
    print("\n开始训练模型...")
    trainer = train_model(config)
    
    print("\n训练完成！")
    if hasattr(trainer, 'best_epoch') and hasattr(trainer, 'best_score'):
        print(f"- 最佳模型在第 {trainer.best_epoch} 轮")
        print(f"- 最佳准确率: {trainer.best_score:.2f}%")
    
    # 可以在这里添加模型评估或测试的代码
    # evaluate_model(trainer.model, config['data']['test_dataset'])


if __name__ == "__main__":
    main()