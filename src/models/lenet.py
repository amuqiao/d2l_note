import torch
from torch import nn
from src.utils.model_registry import ModelRegistry
from src.utils.network_utils import NetworkUtils

# 定义模型默认配置
LENET_CONFIGS = {
    "input_size": (1, 1, 28, 28),  # (batch, channels, height, width)
    "resize": None,                # Fashion-MNIST原始尺寸28x28，无需Resize
    "lr": 0.8,                     # LeNet适合稍高学习率
    "batch_size": 256,             # 较小输入尺寸可支持更大批次
    "num_epochs": 10               # 收敛较快
}

LENET_BATCHNORM_CONFIGS = {
    "input_size": (1, 1, 28, 28),  # (batch, channels, height, width)
    "resize": None,                # Fashion-MNIST原始尺寸28x28，无需Resize
    "lr": 1.0,                     # 带BatchNorm的LeNet学习率
    "batch_size": 256,             # 较小输入尺寸可支持更大批次
    "num_epochs": 2               # 带BatchNorm通常收敛更快
}

@ModelRegistry.register_model("LeNet", LENET_CONFIGS)
class LeNet(nn.Module):
    """LeNet卷积神经网络"""

    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 卷积块2：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),  # 10分类（Fashion-MNIST）
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x

@ModelRegistry.register_model("LeNetBatchNorm", LENET_BATCHNORM_CONFIGS)
class LeNetBatchNorm(nn.Module):
    """带Batch Normalization的LeNet卷积神经网络"""

    def __init__(self):
        super(LeNetBatchNorm, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1：Conv2d → BatchNorm2d → Sigmoid → AvgPool2d
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.BatchNorm2d(6),  # 添加BatchNorm层
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 卷积块2：Conv2d → BatchNorm2d → Sigmoid → AvgPool2d
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),  # 添加BatchNorm层
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),  # 添加BatchNorm层
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  # 添加BatchNorm层
            nn.Sigmoid(),
            nn.Linear(84, 10),  # 10分类（Fashion-MNIST）
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_lenet_shape(net, input_size=(1, 1, 28, 28)):
    """专门用于测试LeNet系列模型的网络形状"""
    NetworkUtils.test_network_shape(net, input_size)

# 注册模型专用测试函数
ModelRegistry.register_test_func("LeNet", test_lenet_shape)
ModelRegistry.register_test_func("LeNetBatchNorm", test_lenet_shape)
