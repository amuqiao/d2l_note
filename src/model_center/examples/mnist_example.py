"""
MNIST 训练示例：展示如何使用 model_center 包进行模型注册和训练

本示例演示了：
1. 如何注册自定义模型
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from src.model_center import ModelRegistry, DataLoaderRegistry, ConfigRegistry, train_model


# ===================== 注册自定义模型 =====================

@ModelRegistry.register(name="simple_cnn", namespace="mnist")
class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型，用于MNIST分类"""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层 -> 激活 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x


@ModelRegistry.register(name="mlp", namespace="mnist")
class MLP(nn.Module):
    """多层感知器模型，用于MNIST分类"""
    def __init__(self, input_size: int = 28*28, hidden_size: int = 128, num_classes: int = 10, dropout_rate: float = 0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        # 展平输入
        x = x.view(-1, 28*28)
        # 全连接层 -> 激活 -> Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x


# ===================== 注册数据集 =====================

# 数据预处理转换
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@DataLoaderRegistry.register_dataset(dataset_name="mnist", namespace="datasets", config={
    "root": "./data",
    "download": True,
    "transform": mnist_transform
})
class MNISTDataset(Dataset):
    """MNIST数据集包装类"""
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


# 数据集预处理函数示例
# 这里可以添加自定义的数据增强或预处理逻辑
# def mnist_preprocess(dataset):
#     # 实现自定义的预处理逻辑
#     return dataset

# 注册预处理函数
# DataLoaderRegistry.register_preprocess_func("mnist", mnist_preprocess, namespace="datasets")


# ===================== 注册配置模板 =====================

# 注册默认MNIST训练配置
ConfigRegistry.register_template("mnist_default", {
    "model": {
        "name": "simple_cnn",
        "namespace": "mnist",
        "args": {
            "num_classes": 10,
            "dropout_rate": 0.5
        }
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
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 2
    },
    "training": {
        "optimizer": {
            "name": "adam",
            "args": {
                "lr": 0.001
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
        "checkpoint_dir": "./checkpoints/mnist",
        "log_interval": 100,
        "early_stopping": {
            "enabled": True,
            "patience": 3
        }
    }
})


# 注册使用MLP模型的配置
ConfigRegistry.register_template("mnist_mlp", {
    "model": {
        "name": "mlp",
        "namespace": "mnist",
        "args": {
            "input_size": 28*28,
            "hidden_size": 128,
            "num_classes": 10,
            "dropout_rate": 0.5
        }
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
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 2
    },
    "training": {
        "optimizer": {
            "name": "sgd",
            "args": {
                "lr": 0.01,
                "momentum": 0.9
            }
        },
        "scheduler": {
            "name": "exponential",
            "args": {
                "gamma": 0.95
            }
        },
        "loss_function": {
            "name": "cross_entropy",
            "args": {}
        },
        "epochs": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints/mnist_mlp",
        "log_interval": 100,
        "early_stopping": {
            "enabled": True,
            "patience": 3
        }
    }
})


# ===================== 主函数 =====================

def main():
    """主函数：加载配置并训练模型"""
    # 选择要使用的配置
    # 可以使用预定义的配置模板
    config = ConfigRegistry.get_config("mnist_default")
    
    # 或者加载自定义的配置文件
    # config = ConfigRegistry.load_config_from_file("path/to/your/config.yaml")
    
    # 也可以动态修改配置
    # config['training']['epochs'] = 5
    # config['model']['name'] = 'mlp'
    
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
    print(f"- 最佳模型在第 {trainer.best_epoch} 轮")
    print(f"- 最佳准确率: {trainer.best_score:.2f}%")


if __name__ == "__main__":
    main()