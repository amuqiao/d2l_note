from torch import nn
from src.utils.model_registry import ModelRegistry


# 定义模型默认配置
MLP_CONFIGS = {
    "input_size": (1, 1, 28, 28),  # (batch, channels, height, width)
    "resize": None,                # Fashion-MNIST原始尺寸28x28，无需Resize
    "lr": 0.5,                     # MLP适合中等学习率
    "batch_size": 256,             # 较大批次可加速训练
    "num_epochs": 10               # 训练轮次
}


@ModelRegistry.register_model("MLP", MLP_CONFIGS)
class MLP(nn.Module):
    """多层感知器(MLP)模型实现
    
    一个简单的全连接神经网络，用于分类任务。
    """

    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入特征维度（默认28*28=784，对应MNIST图像）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（默认10，对应Fashion-MNIST的10个类别）
        """
        super(MLP, self).__init__()
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 添加Dropout防止过拟合
            prev_dim = dim
        
        # 构建分类器（输出层）
        classifier = [
            nn.Linear(prev_dim, output_dim)
        ]
        
        # 将网络分为特征提取和分类器两部分
        self.features = nn.Sequential(
            nn.Flatten(),  # 展平输入
            *layers        # 添加所有隐藏层
        )
        
        self.classifier = nn.Sequential(*classifier)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为[batch_size, 1, 28, 28]
        
        Returns:
            输出张量，形状为[batch_size, 10]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# 演示如何在模型文件中直接测试网络结构
if __name__ == "__main__":
    # 创建模型实例
    net = MLP()
    
    # 创建随机测试输入
    import torch
    X = torch.rand(size=(1, 1, 28, 28))
    
    print("MLP 网络结构测试:")
    print(f"输入形状: {X.shape}")
    
    # 打印特征提取部分每层输出形状
    print("\n特征提取部分:")
    for name, layer in net.features.named_children():
        X = layer(X)
        print(f"{name:<10}: {X.shape}")
    
    # 打印分类器部分每层输出形状
    print("\n分类器部分:")
    for name, layer in net.classifier.named_children():
        X = layer(X)
        print(f"{name:<10}: {X.shape}")
    
    print(f"\n最终输出形状: {X.shape}")