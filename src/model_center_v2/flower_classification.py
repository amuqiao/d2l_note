import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, Any, Tuple, Optional, Callable

# 导入注册中心相关模块
from registry_center import DataLoaderRegistry, ModelRegistry, ConfigRegistry, TrainerRegistry, PredictorRegistry

# 初始化各个注册中心
data_loader_registry = DataLoaderRegistry()
model_registry = ModelRegistry()
config_registry = ConfigRegistry()
trainer_registry = TrainerRegistry()
predictor_registry = PredictorRegistry()

# 数据预处理配置
@config_registry.register_config("flower", "data_transforms")
def get_data_transforms() -> Dict[str, transforms.Compose]:
    """获取数据预处理转换"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# 训练配置
@config_registry.register_config("flower", "train_config")
def get_train_config() -> Dict[str, Any]:
    """获取训练配置参数"""
    return {
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'momentum': 0.9,
        'feature_extract_lr': 0.001,  # 特征提取学习率
        'fine_tune_lr': 0.0001,       # 微调学习率，通常更小
        'feature_extract_unfreeze_layers': 2,  # 微调时解冻的层数
        'data_dir': './flower_data',  # 数据集路径
        'save_dir': './models',       # 模型保存路径
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

# 数据加载器
@data_loader_registry.register_data_loader("flower", "flower_dataloader")
def get_flower_dataloader(data_dir: str, data_transforms: Dict[str, transforms.Compose], batch_size: int) -> Dict[str, DataLoader]:
    """获取花卉数据集的数据加载器"""
    # 创建数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    # 创建数据加载器
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
    
    return {
        'dataloaders': dataloaders_dict,
        'dataset_sizes': {x: len(image_datasets[x]) for x in ['train', 'val']},
        'class_names': image_datasets['train'].classes
    }

# 特征提取模型 (冻结基础层)
@model_registry.register_model("flower", "resnet18", "feature_extract")
class ResNet18FeatureExtract(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet18FeatureExtract, self).__init__()
        # 加载预训练的ResNet18
        self.base_model = models.resnet18(pretrained=True)
        
        # 冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 获取最后一层输入特征数
        num_ftrs = self.base_model.fc.in_features
        
        # 替换最后一层以适应新的分类任务
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

# 微调模型 (解冻部分层)
@model_registry.register_model("flower", "resnet18", "fine_tune")
class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes: int, unfreeze_layers: int = 2):
        super(ResNet18FineTune, self).__init__()
        # 加载预训练的ResNet18
        self.base_model = models.resnet18(pretrained=True)
        
        # 先冻结所有层
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 解冻最后几层
        # ResNet18的layer4包含最后两个卷积块
        layers_to_unfreeze = list(self.base_model.layer4.parameters())
        # 如果需要解冻更多层，可以扩展
        if unfreeze_layers > 1:
            layers_to_unfreeze.extend(list(self.base_model.layer3.parameters()))
        if unfreeze_layers > 2:
            layers_to_unfreeze.extend(list(self.base_model.layer2.parameters()))
        
        for param in layers_to_unfreeze:
            param.requires_grad = True
        
        # 获取最后一层输入特征数
        num_ftrs = self.base_model.fc.in_features
        
        # 替换最后一层以适应新的分类任务
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

# 训练器
@trainer_registry.register_trainer("flower", "resnet_trainer")
class ResNetTrainer:
    def __init__(self, model: nn.Module, dataloaders: Dict[str, DataLoader], dataset_sizes: Dict[str, int],
                 device: torch.device, num_epochs: int = 10, learning_rate: float = 0.001, momentum: float = 0.9):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.num_epochs = num_epochs
        
        # 只优化需要梯度的参数
        params_to_update = self.model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        # 记录训练过程
        self.best_model_wts = None
        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, phase: str) -> Tuple[float, float]:
        """训练单个epoch"""
        if phase == 'train':
            self.model.train()  # 训练模式
        else:
            self.model.eval()   # 评估模式

        running_loss = 0.0
        running_corrects = 0

        # 迭代数据
        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # 只有在训练阶段才进行反向传播和优化
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            self.scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # 记录历史数据
        self.history[f'{phase}_loss'].append(epoch_loss)
        self.history[f'{phase}_acc'].append(epoch_acc.item())

        # 如果是验证阶段且当前模型更好，则保存权重
        if phase == 'val' and epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_model_wts = self.model.state_dict()

        return epoch_loss, epoch_acc.item()
    
    def train(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """完整训练过程"""
        since = time.time()

        # 初始化最佳模型权重
        self.best_model_wts = self.model.state_dict()
        self.best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            # 每个epoch都有训练和验证阶段
            self.train_epoch('train')
            self.train_epoch('val')

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_acc:4f}')

        # 加载最佳模型权重
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.history

# 预测器
@predictor_registry.register_predictor("flower", "resnet_predictor")
class ResNetPredictor:
    def __init__(self, model: nn.Module, device: torch.device, class_names: list):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()  # 设置为评估模式
    
    def predict(self, image: torch.Tensor) -> Tuple[str, float]:
        """预测单张图像"""
        image = image.unsqueeze(0)  # 添加批次维度
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0, preds[0]].item()
        
        return self.class_names[preds[0]], confidence
    
    def visualize_predictions(self, dataloader: DataLoader, num_images: int = 6) -> None:
        """可视化预测结果"""
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure(figsize=(15, 10))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'预测: {self.class_names[preds[j]]}, 实际: {self.class_names[labels[j]]}')
                    
                    # 反归一化图像以便显示
                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        plt.tight_layout()
                        plt.show()
                        return
            self.model.train(mode=was_training)

# 可视化训练历史
def plot_training_history(history: Dict[str, Any], title: str) -> None:
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{title} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title(f'{title} - 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 创建保存模型的目录
    os.makedirs('./models', exist_ok=True)
    
    # 获取配置
    data_transforms = config_registry.get_config("flower", "data_transforms")
    train_config = config_registry.get_config("flower", "train_config")
    
    # 获取数据加载器
    data_loader = data_loader_registry.get_data_loader("flower", "flower_dataloader")
    data = data_loader(
        data_dir=train_config['data_dir'],
        data_transforms=data_transforms,
        batch_size=train_config['batch_size']
    )
    dataloaders = data['dataloaders']
    dataset_sizes = data['dataset_sizes']
    class_names = data['class_names']
    num_classes = len(class_names)
    
    print(f"类别数量: {num_classes}")
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    print(f"使用设备: {train_config['device']}")
    
    # 训练两种模型并比较结果
    results = {}
    
    # 1. 特征提取模型 (冻结基础层)
    print("\n===== 训练特征提取模型 =====")
    feature_extract_model = model_registry.get_model("flower", "resnet18", "feature_extract")(num_classes)
    feature_extract_model = feature_extract_model.to(train_config['device'])
    
    feature_trainer = trainer_registry.get_trainer("flower", "resnet_trainer")(
        model=feature_extract_model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=train_config['device'],
        num_epochs=train_config['num_epochs'],
        learning_rate=train_config['feature_extract_lr'],
        momentum=train_config['momentum']
    )
    
    best_feature_model, feature_history = feature_trainer.train()
    results['feature_extract'] = {
        'model': best_feature_model,
        'history': feature_history,
        'best_acc': max(feature_history['val_acc'])
    }
    
    # 保存模型
    torch.save(best_feature_model.state_dict(), os.path.join(train_config['save_dir'], 'resnet18_feature_extract.pth'))
    
    # 绘制训练历史
    plot_training_history(feature_history, '特征提取模型')
    
    # 2. 微调模型 (解冻部分层)
    print("\n===== 训练微调模型 =====")
    fine_tune_model = model_registry.get_model("flower", "resnet18", "fine_tune")(
        num_classes, 
        unfreeze_layers=train_config['feature_extract_unfreeze_layers']
    )
    fine_tune_model = fine_tune_model.to(train_config['device'])
    
    fine_tune_trainer = trainer_registry.get_trainer("flower", "resnet_trainer")(
        model=fine_tune_model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=train_config['device'],
        num_epochs=train_config['num_epochs'],
        learning_rate=train_config['fine_tune_lr'],
        momentum=train_config['momentum']
    )
    
    best_fine_model, fine_history = fine_tune_trainer.train()
    results['fine_tune'] = {
        'model': best_fine_model,
        'history': fine_history,
        'best_acc': max(fine_history['val_acc'])
    }
    
    # 保存模型
    torch.save(best_fine_model.state_dict(), os.path.join(train_config['save_dir'], 'resnet18_fine_tune.pth'))
    
    # 绘制训练历史
    plot_training_history(fine_history, '微调模型')
    
    # 比较两种方法的结果
    print("\n===== 模型比较 =====")
    print(f"特征提取模型最佳验证准确率: {results['feature_extract']['best_acc']:.4f}")
    print(f"微调模型最佳验证准确率: {results['fine_tune']['best_acc']:.4f}")
    
    # 使用表现更好的模型进行预测可视化
    best_model_type = 'feature_extract' if results['feature_extract']['best_acc'] > results['fine_tune']['best_acc'] else 'fine_tune'
    print(f"\n使用{('特征提取' if best_model_type == 'feature_extract' else '微调')}模型进行预测可视化")
    
    predictor = predictor_registry.get_predictor("flower", "resnet_predictor")(
        model=results[best_model_type]['model'],
        device=train_config['device'],
        class_names=class_names
    )
    
    predictor.visualize_predictions(dataloaders, num_images=6)

if __name__ == "__main__":
    main()
