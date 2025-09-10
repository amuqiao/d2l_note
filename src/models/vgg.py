import torch
from torch import nn

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