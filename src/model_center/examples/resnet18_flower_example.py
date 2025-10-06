"""
ResNet18 迁移学习示例：展示如何使用 model_center 包进行 ResNet18 模型的迁移学习

本示例演示了：
1. 如何使用预训练的 ResNet18 模型进行迁移学习
2. 如何注册自定义模型和数据集
3. 如何创建和加载训练配置
4. 如何使用训练器进行模型训练
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from src.model_center import ModelRegistry, DataLoaderRegistry, ConfigRegistry, train_model


# ===================== 注册 ResNet18 迁移学习模型 =====================

@ModelRegistry.register(name="resnet18_transfer", namespace="flowers", type="cnn")
class ResNet18TransferLearning(nn.Module):
    """基于 ResNet18 的迁移学习模型，用于花卉分类"""
    def __init__(self, num_classes: int = 102, freeze_base: bool = True):
        super(ResNet18TransferLearning, self).__init__()
        
        # 加载预训练的 ResNet18 模型
        self.resnet18 = models.resnet18(pretrained=True)
        
        # 冻结或解冻基础层
        if freeze_base:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        
        # 获取最后一个全连接层的输入特征数
        in_features = self.resnet18.fc.in_features
        
        # 替换最后一个全连接层，用于花卉分类
        self.resnet18.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)


@ModelRegistry.register(name="resnet18_finetune", namespace="flowers", type="cnn")
class ResNet18FineTuning(nn.Module):
    """基于 ResNet18 的微调模型，用于花卉分类"""
    def __init__(self, num_classes: int = 102, unfreeze_layers: int = 2):
        super(ResNet18FineTuning, self).__init__()
        
        # 加载预训练的 ResNet18 模型
        self.resnet18 = models.resnet18(pretrained=True)
        
        # 默认先冻结所有层
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        # 根据需要解冻特定层（通常是最后几个卷积层组）
        # ResNet18 的层结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        if unfreeze_layers >= 1:
            for param in self.resnet18.layer4.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 2:
            for param in self.resnet18.layer3.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 3:
            for param in self.resnet18.layer2.parameters():
                param.requires_grad = True
        if unfreeze_layers >= 4:
            for param in self.resnet18.layer1.parameters():
                param.requires_grad = True
        
        # 获取最后一个全连接层的输入特征数
        in_features = self.resnet18.fc.in_features
        
        # 替换最后一个全连接层，用于花卉分类
        self.resnet18.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet18(x)


# ===================== 注册 Flower 数据集 =====================

# 数据预处理转换 - 训练集（包含数据增强）
flower_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 的均值和标准差
])

# 数据预处理转换 - 测试集
flower_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 的均值和标准差
])


@DataLoaderRegistry.register_dataset(dataset_name="flowers", namespace="datasets", config={
    "root": "./data/flowers",
    "download": True
})
class FlowerDataset(Dataset):
    """Flower 数据集包装类"""
    def __init__(self, root: str = "./data/flowers", train: bool = True, download: bool = True, transform=None):
        # 使用 torchvision 的 ImageFolder 数据集加载花卉数据
        # 注意：这里假设花卉数据已经按照类别组织在子文件夹中
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# 自定义数据集下载和准备函数
def prepare_flower_dataset(root: str = "./data/flowers"):
    """准备花卉数据集，如果不存在则下载和解压"""
    import os
    # 确保root是绝对路径
    root = os.path.abspath(root)
    
    # 检查是否已处理过数据集
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
        print(f"花卉数据集已处理完成，训练数据位于 '{train_dir}'")
        return
    
    # 检查数据集文件是否存在
    images_tgz = os.path.join(root, "102flowers.tgz")
    labels_mat = os.path.join(root, "imagelabels.mat")
    setid_mat = os.path.join(root, "setid.mat")
    
    # 创建数据目录
    os.makedirs(root, exist_ok=True)
    
    # 下载数据集文件（如果不存在）
    if not (os.path.exists(images_tgz) and os.path.exists(labels_mat) and os.path.exists(setid_mat)):
        print("花卉数据集文件不存在，正在下载...")
        
        import urllib.request
        
        # 数据集URL
        dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
        setid_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
        
        try:
            # 下载数据集文件
            print("下载数据集图像...")
            urllib.request.urlretrieve(dataset_url, images_tgz)
            print("下载数据集标签...")
            urllib.request.urlretrieve(labels_url, labels_mat)
            print("下载数据集分割信息...")
            urllib.request.urlretrieve(setid_url, setid_mat)
        except Exception as e:
            print(f"下载数据集时出错: {e}")
            print(f"请手动下载花卉数据集并将文件放在 '{root}' 目录中。")
            return
    
    # 解压和处理数据集
    print("数据集下载完成，正在解压和准备数据...")
    
    try:
        # 解压tgz文件和处理数据
        import tarfile
        import scipy.io
        import shutil
        import os
        
        # 解压图像文件
        images_dir = os.path.join(root, "images")
        if not os.path.exists(images_dir):
            print("解压图像文件...")
            with tarfile.open(images_tgz, 'r:gz') as tar:
                tar.extractall(path=root)
            # 重命名解压后的目录
            extracted_dir = os.path.join(root, "jpg")
            if os.path.exists(extracted_dir):
                os.rename(extracted_dir, images_dir)
        
        # 读取标签和数据集分割信息
        print("读取标签和数据集分割信息...")
        labels_data = scipy.io.loadmat(labels_mat)
        setid_data = scipy.io.loadmat(setid_mat)
        
        # 获取标签和分割信息
        labels = labels_data['labels'][0]  # 标签从1开始
        train_ids = setid_data['trnid'][0]  # 训练集索引
        val_ids = setid_data['valid'][0]  # 验证集索引
        test_ids = setid_data['tstid'][0]  # 测试集索引
        
        # 创建训练、验证和测试目录
        print("创建数据集目录结构...")
        for split, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            split_dir = os.path.join(root, split)
            
            # 为每个类别创建子目录
            for i in range(1, 103):  # 102个类别
                class_dir = os.path.join(split_dir, str(i))
                os.makedirs(class_dir, exist_ok=True)
            
            # 复制图像到对应的类别目录
            for img_id in ids:
                src_path = os.path.join(images_dir, f"image_{img_id:05d}.jpg")
                if os.path.exists(src_path):
                    class_label = labels[img_id - 1]  # 索引从0开始
                    dst_dir = os.path.join(split_dir, str(class_label))
                    dst_path = os.path.join(dst_dir, f"image_{img_id:05d}.jpg")
                    # 使用复制而不是移动，保留原始图像
                    shutil.copy(src_path, dst_path)
        
        print("花卉数据集准备完成！")
        print(f"- 训练数据: {train_dir}")
        print(f"- 验证数据: {val_dir}")
        print(f"- 测试数据: {test_dir}")
        
    except Exception as e:
        print(f"处理数据集时出错: {e}")
        print("请手动处理数据集文件。")
        return


# ===================== 注册配置模板 =====================

# 注册默认的 ResNet18 迁移学习配置
ConfigRegistry.register_template("resnet18_flower_transfer", {
    "model": {
        "name": "resnet18_transfer",
        "namespace": "flowers",
        "type": "cnn",
        "args": {
            "num_classes": 102,
            "freeze_base": True
        }
    },
    "data": {
        "train_dataset": {
            "name": "flowers",
            "namespace": "datasets",
            "args": {
                "root": "./data/flowers/train",
                "transform": flower_train_transform
            }
        },
        "test_dataset": {
            "name": "flowers",
            "namespace": "datasets",
            "args": {
                "root": "./data/flowers/val",
                "transform": flower_test_transform
            }
        },
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4
    },
    "training": {
        "optimizer": {
            "name": "adam",
            "args": {
                "lr": 0.001,
                "weight_decay": 0.0001
            }
        },
        "scheduler": {
            "name": "reduce_lr_on_plateau",
            "args": {
                "factor": 0.1,
                "patience": 3
            }
        },
        "loss_function": {
            "name": "cross_entropy",
            "args": {}
        },
        "epochs": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints/resnet18_flower_transfer",
        "log_interval": 50,
        "early_stopping": {
            "enabled": True,
            "patience": 5
        }
    }
})


# 注册 ResNet18 微调配置
ConfigRegistry.register_template("resnet18_flower_finetune", {
    "model": {
        "name": "resnet18_finetune",
        "namespace": "flowers",
        "type": "cnn",
        "args": {
            "num_classes": 102,
            "unfreeze_layers": 2
        }
    },
    "data": {
        "train_dataset": {
            "name": "flowers",
            "namespace": "datasets",
            "args": {
                "root": "./data/flowers/train",
                "transform": flower_train_transform
            }
        },
        "test_dataset": {
            "name": "flowers",
            "namespace": "datasets",
            "args": {
                "root": "./data/flowers/val",
                "transform": flower_test_transform
            }
        },
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4
    },
    "training": {
        "optimizer": {
            "name": "sgd",
            "args": {
                "lr": 0.0001,  # 微调时学习率要小一些
                "momentum": 0.9,
                "weight_decay": 0.0001
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
        "epochs": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints/resnet18_flower_finetune",
        "log_interval": 50,
        "early_stopping": {
            "enabled": True,
            "patience": 5
        }
    }
})


# ===================== 主函数 =====================

def main():
    """主函数：加载配置并训练模型"""
    # 准备花卉数据集
    prepare_flower_dataset()
    
    # 选择要使用的配置
    # 可以使用迁移学习配置
    config = ConfigRegistry.get_config("resnet18_flower_transfer")
    
    # 也可以使用微调配置
    # config = ConfigRegistry.get_config("resnet18_flower_finetune")
    
    # 或者动态修改配置
    # config['training']['epochs'] = 10
    # config['data']['batch_size'] = 64
    
    # 打印配置信息
    print("\n训练配置:")
    print(f"- 模型: {config['model']['name']}")
    print(f"- 设备: {config['training']['device']}")
    print(f"- 批次大小: {config['data']['batch_size']}")
    print(f"- 训练轮数: {config['training']['epochs']}")
    print(f"- 优化器: {config['training']['optimizer']['name']}")
    print(f"- 学习率: {config['training']['optimizer']['args']['lr']}")
    print(f"- 检查点目录: {config['training']['checkpoint_dir']}")
    
    # 训练模型
    print("\n开始训练模型...")
    trainer = train_model(config)
    
    print("\n训练完成！")
    if hasattr(trainer, 'best_epoch') and hasattr(trainer, 'best_score'):
        print(f"- 最佳模型在第 {trainer.best_epoch} 轮")
        print(f"- 最佳准确率: {trainer.best_score:.2f}%")


if __name__ == "__main__":
    main()