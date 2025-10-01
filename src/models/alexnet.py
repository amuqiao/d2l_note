import torch
from torch import nn
from src.utils.model_registry import ModelRegistry
from src.utils.network_utils import NetworkUtils


# 定义模型默认配置
ALEXNET_CONFIGS = {
    "input_size": (1, 1, 224, 224), # AlexNet需要224x224输入
    "resize": 224,                  # 加载数据时Resize到224x224
    "lr": 0.01,                     # 较大模型需较低学习率避免震荡
    "batch_size": 128,              # 224x224输入占用显存较高，批次减小
    "num_epochs": 30                # 训练较慢，30轮平衡效果与时间
}


@ModelRegistry.register_model("AlexNet", ALEXNET_CONFIGS)
class AlexNet(nn.Module):
    """AlexNet卷积神经网络"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 使用dropout层来减轻过拟合
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_alexnet_shape(net, input_size=(1, 1, 224, 224)):
    """专门用于测试AlexNet模型的网络形状"""
    NetworkUtils.test_network_shape(net, input_size)

# 注册模型专用测试函数
ModelRegistry.register_test_func("AlexNet", test_alexnet_shape)
