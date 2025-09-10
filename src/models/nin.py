import torch
from torch import nn

class NIN(nn.Module):
    """Network in Network (NIN)卷积神经网络"""
    
    def __init__(self):
        super(NIN, self).__init__()
        # 定义NIN块构造函数
        def nin_block(in_channels, out_channels, kernel_size, strides, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
        
        # 构建网络特征提取层
        self.features = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            # 标签类别数是10
            nin_block(384, 10, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器层：简单的展平操作
        self.classifier = nn.Sequential(
            nn.Flatten()  # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x