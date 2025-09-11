import torch
from torch import nn
import torch.nn.functional as F
from src.utils.model_registry import ModelRegistry
from src.utils.network_utils import NetworkUtils


# 定义模型默认配置
GOOGLENET_CONFIGS = {
    "input_size": (1, 1, 96, 96),   # GoogLeNet需要96x96输入
    "resize": 96,                   # 加载数据时Resize到96x96
    "lr": 0.1,                      # 学习率设置
    "batch_size": 128,              # 批次大小设置
    "num_epochs": 20                # 训练轮次设置
}


@ModelRegistry.register_model("GoogLeNet", GOOGLENET_CONFIGS)
class GoogLeNet(nn.Module):
    """GoogLeNet卷积神经网络"""
    
    def __init__(self):
        super(GoogLeNet, self).__init__()
        
        # 构建网络特征提取层
        # 第一模块：7×7卷积层
        b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第二模块：1×1卷积层后接3×3卷积层
        b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第三模块：两个Inception块
        b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第四模块：五个Inception块
        b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第五模块：两个Inception块+全局平均池化
        b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 组合所有模块
        self.features = nn.Sequential(b1, b2, b3, b4, b5)
        
        # 分类器层：全连接层
        self.classifier = nn.Sequential(
            nn.Linear(1024, 10)  # 10分类（Fashion-MNIST）
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


class Inception(nn.Module):
    """Inception模块实现"""
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


def test_googlenet_shape(net, input_size=(1, 1, 96, 96)):
    """专门用于测试GoogLeNet模型的网络形状"""
    NetworkUtils.test_network_shape(net, input_size)

# 注册模型专用测试函数
ModelRegistry.register_test_func("GoogLeNet", test_googlenet_shape)