import torch
from torch import nn
from src.utils.model_registry import ModelRegistry
from src.utils.network_utils import NetworkUtils


# 定义模型默认配置
VGG_CONFIGS = {
    "input_size": (1, 1, 224, 224), # VGG同样需要224x224输入
    "resize": 224,                  # 加载数据时Resize到224x224
    "lr": 0.05,                     # 更深模型需更低学习率
    "batch_size": 128,              # VGG参数量大，显存占用更高
    "num_epochs": 10                # 训练耗时久，10轮兼顾效果
}


@ModelRegistry.register_model("VGG", VGG_CONFIGS)
class VGG(nn.Module):
    """VGG卷积神经网络（使用简化版本适配Fashion-MNIST）"""
    
    def __init__(self, ratio=4):
        super(VGG, self).__init__()
        # 定义VGG块构造函数
        def vgg_block(num_convs, in_channels, out_channels):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)
        
        # VGG配置 - 标准版本
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        
        # 简化版本：使用比例因子缩小通道数
        small_conv_arch = [(num_convs, out_channels // ratio) 
                          for (num_convs, out_channels) in conv_arch]
        
        # 卷积层部分
        conv_blks = []
        in_channels = 1
        for (num_convs, out_channels) in small_conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        
        # 构建网络
        self.features = nn.Sequential(*conv_blks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 全连接层部分
            nn.Linear(in_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


def test_vgg_shape(net, input_size=(1, 1, 224, 224)):
    """专门用于测试VGG模型的网络形状"""
    NetworkUtils.test_network_shape(net, input_size)

# 注册模型专用测试函数
ModelRegistry.register_test_func("VGG", test_vgg_shape)