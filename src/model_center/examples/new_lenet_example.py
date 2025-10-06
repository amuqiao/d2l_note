"""
LeNet 训练示例：使用统一注册入口重新实现

本示例演示了：
1. 如何使用统一注册入口注册和创建 LeNet 模型
2. 如何使用统一注册入口注册和创建数据集
3. 如何使用统一注册入口注册和加载训练配置
4. 如何使用训练器进行模型训练
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 使用统一注册入口
from src.model_center.model_registry_entry import register_model, create_model, get_model_config
from src.model_center.data_loader_registry_entry import register_dataset, create_train_test_loaders
from src.model_center.config_registry_entry import register_template, get_config

# 从 src/models/lenet.py 导入 LeNet 和 LeNetBatchNorm 模型
from src.models.lenet import LeNet, LeNetBatchNorm

# ===================== 使用统一注册入口注册 LeNet 模型 =====================

@register_model(name="LeNet", namespace="lenet", type="cnn", config={
    "num_classes": 10,
    "lr": 0.8,  # LeNet 适合稍高学习率
    "batch_size": 256
})
class RegisteredLeNet(LeNet):
    """使用统一注册入口注册的 LeNet 模型"""
    pass


@register_model(name="LeNetBatchNorm", namespace="lenet", type="cnn", config={
    "num_classes": 10,
    "lr": 1.0,  # 带 BatchNorm 的 LeNet 学习率
    "batch_size": 256
})
class RegisteredLeNetBatchNorm(LeNetBatchNorm):
    """使用统一注册入口注册的带 Batch Normalization 的 LeNet 模型"""
    pass


# ===================== 使用统一注册入口注册数据集 =====================

# 数据预处理转换
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
])


@register_dataset(name="mnist", namespace="datasets", config={
    "root": "./data",
    "download": True,
    "transform": mnist_transform
})
class MNISTDataset:
    """MNIST 数据集包装类"""
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform=None):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# ===================== 使用统一注册入口注册配置模板 =====================

# 注册默认 LeNet 训练配置
register_template(
    config_name="lenet_default",
    template={
        "model": {
            "name": "LeNet",
            "namespace": "lenet",
            "type": "cnn",
            "args": {}
        },
        "data": {
            "train_dataset": {
                "name": "mnist",
                "namespace": "datasets",
                "args": {
                    "train": True
                }
            },
            "test_dataset": {
                "name": "mnist",
                "namespace": "datasets",
                "args": {
                    "train": False
                }
            },
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2
        },
        "training": {
            "optimizer": {
                "name": "sgd",
                "args": {
                    "lr": 0.8,
                    "momentum": 0.9
                }
            },
            "scheduler": {
                "name": "step",
                "args": {
                    "step_size": 5,
                    "gamma": 0.1
                }
            },
            "loss_function": {
                "name": "cross_entropy",
                "args": {}
            },
            "epochs": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "checkpoint_dir": "./checkpoints/lenet",
            "log_interval": 100,
            "early_stopping": {
                "enabled": True,
                "patience": 3
            }
        }
    },
    namespace="default"
)


# 注册 LeNetBatchNorm 训练配置
register_template(
    config_name="lenet_batchnorm",
    template={
        "model": {
            "name": "LeNetBatchNorm",
            "namespace": "lenet",
            "type": "cnn",
            "args": {}
        },
        "data": {
            "train_dataset": {
                "name": "mnist",
                "namespace": "datasets",
                "args": {
                    "train": True
                }
            },
            "test_dataset": {
                "name": "mnist",
                "namespace": "datasets",
                "args": {
                    "train": False
                }
            },
            "batch_size": 256,
            "shuffle": True,
            "num_workers": 2
        },
        "training": {
            "optimizer": {
                "name": "sgd",
                "args": {
                    "lr": 1.0,
                    "momentum": 0.9
                }
            },
            "scheduler": {
                "name": "step",
                "args": {
                    "step_size": 2,
                    "gamma": 0.1
                }
            },
            "loss_function": {
                "name": "cross_entropy",
                "args": {}
            },
            "epochs": 5,  # 带 BatchNorm 通常收敛更快
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "checkpoint_dir": "./checkpoints/lenet_batchnorm",
            "log_interval": 100,
            "early_stopping": {
                "enabled": True,
                "patience": 2
            }
        }
    },
    namespace="default"
)


# ===================== 主函数 =====================

def main():
    """主函数：使用统一注册入口加载配置并训练模型"""
    # 使用统一注册入口获取配置
    # 可以使用 LeNet 的默认配置
    # config = get_config("lenet_default", namespace="default")
    
    # 也可以使用带 BatchNorm 的配置
    config = get_config("lenet_batchnorm", namespace="default")
    
    # 或者动态修改配置
    # config['training']['epochs'] = 15
    # config['training']['optimizer']['args']['lr'] = 0.01
    
    # 打印配置信息
    print("\n训练配置:")
    print(f"- 模型: {config['model']['name']}")
    print(f"- 设备: {config['training']['device']}")
    print(f"- 批次大小: {config['data']['batch_size']}")
    print(f"- 训练轮数: {config['training']['epochs']}")
    print(f"- 优化器: {config['training']['optimizer']['name']}")
    print(f"- 学习率: {config['training']['optimizer']['args']['lr']}")
    print(f"- 检查点目录: {config['training']['checkpoint_dir']}")
    
    # 使用统一注册入口创建训练和测试数据加载器
    train_loader, test_loader = create_train_test_loaders(
        dataset_name="mnist",
        namespace="datasets",
        train_batch_size=config['data']['batch_size'],
        test_batch_size=config['data']['batch_size'],
        train_shuffle=config['data']['shuffle'],
        test_shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 使用统一注册入口创建模型
    model = create_model(
        model_name=config['model']['name'],
        namespace=config['model']['namespace'],
        type=config['model']['type'],
        **config['model']['args']
    )
    
    # 将模型移至指定设备
    device = torch.device(config['training']['device'])
    model.to(device)
    
    # 初始化优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['args']['lr'],
        momentum=config['training']['optimizer']['args']['momentum']
    )
    
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler']['args']['step_size'],
        gamma=config['training']['scheduler']['args']['gamma']
    )
    
    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建检查点目录
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # 简单训练循环
    best_accuracy = 0.0
    best_epoch = 0
    
    print("\n开始训练模型...")
    for epoch in range(config['training']['epochs']):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 打印训练进度
            if batch_idx % config['training']['log_interval'] == 0:
                print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # 更新学习率
        scheduler.step()
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, os.path.join(config['training']['checkpoint_dir'], 'best_model.pth'))
    
    print("\n训练完成！")
    print(f"- 最佳模型在第 {best_epoch} 轮")
    print(f"- 最佳准确率: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()