import torch
from torch import nn
from torch.nn import functional as F
from src.utils.model_registry import ModelRegistry
from src.utils.network_utils import NetworkUtils


# 定义模型默认配置
RESNET_CONFIGS = {
    "input_size": (1, 1, 224, 224),  # (batch, channels, height, width)
    "resize": 224,                    # 加载数据时Resize到224x224
    "lr": 0.05,                       # 学习率
    "batch_size": 128,                # 批次大小
    "num_epochs": 10                  # 训练轮次
}

class Residual(nn.Module):
    """残差块：ResNet的基本构建单元"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 主路径：第一个卷积层
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 主路径：第二个卷积层
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        # 捷径路径：可选的1x1卷积层（用于通道数变化或空间尺寸变化时）
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)  # 应用捷径路径的1x1卷积
        Y += X  # 残差连接
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """构建残差块序列"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 非第一个块的第一个残差块需要下采样和通道数调整
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 其他残差块保持通道数和空间尺寸不变
            blk.append(Residual(num_channels, num_channels))
    return blk


@ModelRegistry.register_model("ResNet", RESNET_CONFIGS)
class ResNet(nn.Module):
    """ResNet模型实现（ResNet-18）"""

    def __init__(self):
        super(ResNet, self).__init__()
        # 第一层：卷积+批归一化+ReLU+最大池化
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 残差块序列
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化到1x1
            nn.Flatten(),
            nn.Linear(512, 10)  # 10分类（Fashion-MNIST）
        )
        
        # 特征提取路径
        self.features = nn.Sequential(
            self.b1,
            self.b2,
            self.b3,
            self.b4,
            self.b5
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


# 配置已经在文件顶部定义，这里不再重复定义


def test_resnet_shape(net, input_size=(1, 1, 224, 224)):
    """专门用于测试ResNet模型的网络形状"""
    NetworkUtils.test_network_shape(net, input_size)

# 注册模型专用测试函数
ModelRegistry.register_test_func("ResNet", test_resnet_shape)

if __name__ == "__main__":
    # 简单测试模型结构
    net = ResNet()
    X = torch.rand(size=(1, 1, 224, 224))
    print("ResNet 网络结构测试:")
    print(f"输入形状: {X.shape}")
    
    # 打印特征提取部分每层输出形状
    print("\n特征提取部分:")
    for name, layer in net.features.named_children():
        X = layer(X)
        print(f"{name:<4}: {X.shape}")
    
    # 打印分类器部分每层输出形状
    print("\n分类器部分:")
    for name, layer in net.classifier.named_children():
        X = layer(X)
        print(f"{name:<15}: {X.shape}")
    
    print(f"\n最终输出形状: {X.shape}")


