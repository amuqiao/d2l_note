"""网络结构工具类"""
import torch
from torch import nn

class NetworkUtils:
    """网络结构工具类：提供网络结构测试等功能"""

    @staticmethod
    def test_network_shape(net, input_size=(1, 1, 28, 28)):
        """测试网络各层输出形状"""
        # 检查网络是否包含BatchNorm层
        has_batch_norm = NetworkUtils._has_batch_norm(net)
        
        # 保存网络当前状态
        is_training = net.training
        net.eval()  # 切换到评估模式
        
        try:
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
                # 对于BatchNorm网络，使用批次大小为2进行测试
                if has_batch_norm and input_size[0] == 1:
                    test_input_size = (2,) + input_size[1:]  # 临时改为批次大小2
                    
                else:
                    test_input_size = input_size
                
                X = net.features(torch.rand(size=test_input_size, dtype=torch.float32))
                
                if hasattr(net, "classifier"):
                    for i, layer in enumerate(net.classifier):
                        X = layer(X)
                        print(
                            f"   {i+1:2d}. {layer.__class__.__name__:12s} → 输出形状: {X.shape}"
                        )
        finally:
            # 恢复网络原始状态
            if is_training:
                net.train()
    
    @staticmethod
    def _has_batch_norm(net):
        """检查网络是否包含BatchNorm层"""
        for module in net.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False
    
    @staticmethod
    def test_lenet_shape(net, input_size=(1, 1, 28, 28)):
        """专门用于测试LeNet系列模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)
    
    @staticmethod
    def test_alexnet_shape(net, input_size=(1, 1, 224, 224)):
        """专门用于测试AlexNet模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)
    
    @staticmethod
    def test_vgg_shape(net, input_size=(1, 1, 224, 224)):
        """专门用于测试VGG模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)
    
    @staticmethod
    def test_nin_shape(net, input_size=(1, 1, 224, 224)):
        """专门用于测试NIN模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)
    
    @staticmethod
    def test_googlenet_shape(net, input_size=(1, 1, 96, 96)):
        """专门用于测试GoogLeNet模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)
    
    @staticmethod
    def test_resnet_shape(net, input_size=(1, 1, 224, 224)):
        """专门用于测试ResNet模型的网络形状"""
        NetworkUtils.test_network_shape(net, input_size)