"""网络结构工具类"""
import torch
from torch import nn

class NetworkUtils:
    """网络结构工具类：提供网络结构测试等功能"""

    @staticmethod
    def test_network_shape(net, input_size=(1, 1, 28, 28)):
        """测试网络各层输出形状"""
        X = torch.rand(size=input_size, dtype=torch.float32)
        print("\n🔍 网络结构测试:")

        # 如果网络使用Sequential实现
        if isinstance(net, nn.Sequential):
            for i, layer in enumerate(net):
                X = layer(X)
                print(f"{i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}")
        else:
            # 特征提取层
            print("├─ 特征提取部分:")
            if hasattr(net, "features"):
                for i, layer in enumerate(net.features):
                    X = layer(X)
                    print(
                        f"│  {i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}"
                    )

            # 分类器层
            print("└─ 分类器部分:")
            X = net.features(torch.rand(size=input_size, dtype=torch.float32))
            if hasattr(net, "classifier"):
                for i, layer in enumerate(net.classifier):
                    X = layer(X)
                    print(
                        f"   {i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}"
                    )