"""
模型注册入口示例：展示如何使用统一注册入口和自动注册机制

本示例演示了：
1. 使用简化的register_model装饰器自动注册模型
2. 使用统一注册入口的便捷函数创建模型和获取模型信息
3. 如何查看所有已注册的模型
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
# 先导入register_model装饰器来注册模型
from src.model_center.model_registry_entry import register_model

# 然后导入其他需要的功能
from src.model_center import (
    # 新添加的统一注册入口便捷函数
    create_model,
    get_model_config,
    get_all_registered_models,
    is_model_registered
)

# ===================== 使用自动注册装饰器注册模型 =====================

@register_model(name="simple_auto_mlp", namespace="demo", type="mlp", config={
    "input_size": 784,
    "hidden_size": 128,
    "num_classes": 10,
    "dropout_rate": 0.5
})
class SimpleAutoMLP(nn.Module):
    """使用自动注册装饰器注册的MLP模型"""
    def __init__(self, input_size: int = 784, hidden_size: int = 128, 
                 num_classes: int = 10, dropout_rate: float = 0.5):
        super(SimpleAutoMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = x.view(-1, self.fc1.in_features)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

@register_model(name="simple_auto_cnn", namespace="demo", type="cnn")
class SimpleAutoCNN(nn.Module):
    """使用自动注册装饰器注册的CNN模型"""
    def __init__(self, num_classes: int = 10):
        super(SimpleAutoCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 使用统一注册入口的便捷函数 =====================

def demonstrate_model_registry_entry():
    """演示统一注册入口的使用"""
    print("="*60)
    print("演示model_center统一注册入口和自动注册机制")
    print("="*60)
    
    # 1. 检查模型是否已注册
    print("\n1. 检查模型是否已注册：")
    models_to_check = [
        ("simple_auto_mlp", "demo"),
        ("simple_auto_cnn", "demo"),
        ("simple_cnn", "mnist"),  # 这是之前例子中注册的模型
        ("non_existent_model", "default")
    ]
    
    for model_name, namespace in models_to_check:
        is_registered = is_model_registered(model_name, namespace)
        print(f"  - 模型 '{model_name}'（命名空间 '{namespace}'）已注册: {is_registered}")
    
    # 2. 获取所有已注册的模型
    print("\n2. 获取所有已注册的模型：")
    all_models = get_all_registered_models()
    for ns, models in all_models.items():
        print(f"  命名空间 '{ns}':")
        for model in models:
            print(f"    - {model['name']} (类型: {model['type']}, 类: {model['class_name']})")
    
    # 3. 获取模型配置
    print("\n3. 获取模型配置：")
    config = get_model_config("simple_auto_mlp", "demo", type="mlp")
    print(f"  'simple_auto_mlp' 模型配置: {config}")
    
    # 4. 创建模型实例
    print("\n4. 创建模型实例：")
    # 使用默认配置创建模型，注意要指定正确的type参数
    mlp_model = create_model("simple_auto_mlp", "demo", type="mlp")
    print(f"  - 使用默认配置创建MLP模型: {mlp_model.__class__.__name__}")
    
    # 使用自定义配置创建模型
    custom_mlp_model = create_model("simple_auto_mlp", "demo", type="mlp",
                                    hidden_size=256, dropout_rate=0.3)
    print(f"  - 使用自定义配置创建MLP模型: {custom_mlp_model.__class__.__name__}（hidden_size=256, dropout_rate=0.3）")
    
    # 创建CNN模型
    cnn_model = create_model("simple_auto_cnn", "demo", type="cnn", num_classes=10)
    print(f"  - 创建CNN模型: {cnn_model.__class__.__name__}")
    
    # 5. 测试模型前向传播
    print("\n5. 测试模型前向传播：")
    # 为MLP创建随机输入 (batch_size=4, input_size=784)
    mlp_input = torch.randn(4, 784)
    mlp_output = mlp_model(mlp_input)
    print(f"  - MLP输出形状: {mlp_output.shape}")
    
    # 为CNN创建随机输入 (batch_size=4, channels=1, height=28, width=28)
    cnn_input = torch.randn(4, 1, 28, 28)
    cnn_output = cnn_model(cnn_input)
    print(f"  - CNN输出形状: {cnn_output.shape}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)

if __name__ == "__main__":
    # 演示统一注册入口的使用
    demonstrate_model_registry_entry()