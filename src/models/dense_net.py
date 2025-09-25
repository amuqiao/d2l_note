import torch
from torch import nn
from d2l import torch as d2l
from src.utils.model_registry import ModelRegistry


# 模型配置
DENSENET_CONFIGS = {
    "input_size": (1, 1, 96, 96),  # (batch, channels, height, width)
    "resize": 96,                    # 加载数据时Resize到96x96
    "lr": 0.1,                       # 学习率
    "batch_size": 256,               # 批次大小
    "num_epochs": 10                 # 训练轮次
}


def conv_block(input_channels, num_channels):
    """卷积块：DenseNet的基本构建单元"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
    

class DenseBlock(nn.Module):
    """稠密块：由多个卷积块组成，特征图在通道维度上连接"""
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    """转换层：用于稠密块之间的过渡，降低特征图尺寸并调整通道数"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))


@ModelRegistry.register_model("DenseNet", DENSENET_CONFIGS)
class DenseNet(nn.Module):
    """DenseNet模型实现"""

    def __init__(self, growth_rate=32, num_convs_in_dense_blocks=[4, 4, 4, 4]):
        super(DenseNet, self).__init__()
        # 第一层：卷积+批归一化+ReLU+最大池化
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 构建稠密块和转换层序列
        num_channels = 64
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间添加一个转换层，使通道数量减半
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
                
        # 特征提取路径
        self.features = nn.Sequential(
            self.b1,
            *blks,
            nn.BatchNorm2d(num_channels), nn.ReLU()
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化到1x1
            nn.Flatten(),
            nn.Linear(num_channels, 10)  # 10分类（Fashion-MNIST）
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 简单测试模型结构
    net = DenseNet()
    X = torch.rand(size=(1, 1, 96, 96))
    print("DenseNet 网络结构测试:")
    print(f"输入形状: {X.shape}")
    
    # 打印特征提取部分每层输出形状
    print("\n特征提取部分:")
    for name, layer in net.features.named_children():
        if isinstance(layer, nn.Sequential):
            print(f"{name:<4}:")
            for sub_name, sub_layer in layer.named_children():
                X = sub_layer(X)
                print(f"  {sub_name:<15}: {X.shape}")
        else:
            X = layer(X)
            print(f"{name:<4}: {X.shape}")
    
    # 打印分类器部分每层输出形状
    print("\n分类器部分:")
    for name, layer in net.classifier.named_children():
        X = layer(X)
        print(f"{name:<15}: {X.shape}")
    
    print(f"\n最终输出形状: {X.shape}")
    
    # 如需训练模型，可以取消下面的注释
    """
    lr, num_epochs, batch_size = DENSENET_CONFIGS["lr"], DENSENET_CONFIGS["num_epochs"], DENSENET_CONFIGS["batch_size"]
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=DENSENET_CONFIGS["resize"])
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    """

