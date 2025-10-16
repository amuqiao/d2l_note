import os
# 设置环境变量解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # 新增：用于图像处理

# 导入注册中心相关模块
from registry_center import DataLoaderRegistry, ModelRegistry, ConfigRegistry, TrainerRegistry, PredictorRegistry

# 1. 数据加载器实现与注册
class MNISTDataLoader:
    def __init__(self, batch_size=64, data_dir='./data'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def get_train_loader(self):
        train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def get_test_loader(self):
        test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

# 注册数据加载器
data_loader_registry = DataLoaderRegistry()
data_loader_registry.register_data_loader("mnist", "standard", MNISTDataLoader)

# 2. LeNet模型实现与注册
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 注册LeNet模型
model_registry = ModelRegistry()
model_registry.register_model("cnn", "lenet", "classification")(LeNet)

# 3. 训练配置实现与注册
class TrainingConfig:
    def __init__(self):
        self.epochs = 3
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_interval = 100

# 注册训练配置
config_registry = ConfigRegistry()
config_registry.register_config("mnist", "lenet_config", TrainingConfig)

# 4. 训练器实现与注册
class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum
        )
        
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % self.config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        self.train_losses.append(avg_loss)
        self.train_acc.append(accuracy)
        print(f'Train Epoch: {epoch} Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        self.test_losses.append(test_loss)
        self.test_acc.append(accuracy)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    def train(self):
        print(f"Using device: {self.config.device}")
        for epoch in range(1, self.config.epochs + 1):
            self.train_epoch(epoch)
            self.test()
        
        self.plot_results()
    
    def plot_results(self):
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, len(self.test_losses) + 1), self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.train_acc) + 1), self.train_acc, label='Train Accuracy')
        plt.plot(range(1, len(self.test_acc) + 1), self.test_acc, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# 注册训练器
trainer_registry = TrainerRegistry()
trainer_registry.register_trainer("mnist", "standard_trainer", Trainer)

# 创建预测器注册中心实例
predictor_registry = PredictorRegistry()

# 5. 预测器实现与注册
class LeNetPredictor:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.model.eval()  # 切换到评估模式
        self.device = config.device
        # 复用训练时的预处理逻辑
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def predict(self, image):
        """
        预测单张图像的类别
        :param image: 输入图像（PIL Image，灰度图）
        :return: 预测的类别索引和置信度
        """
        # 预处理图像
        image_tensor = self.transform(image).unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():  # 关闭梯度计算
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)  # 转换为概率
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return predicted_class.item(), confidence.item()

# 注册预测器到PredictorRegistry
predictor_registry.register_predictor("mnist", "lenet_predictor")(LeNetPredictor)

# 6. 主函数：从注册中心获取组件并执行训练+预测
def main():
    # 从注册中心获取组件
    data_loader_cls = data_loader_registry.get_data_loader("mnist", "standard")
    model_cls = model_registry.get_model("cnn", "lenet", "classification")
    config_cls = config_registry.get_config("mnist", "lenet_config")
    trainer_cls = trainer_registry.get_trainer("mnist", "standard_trainer")
    
    # 实例化组件
    data_loader = data_loader_cls(batch_size=64)
    model = model_cls()
    config = config_cls()
    
    # 获取数据加载器
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # 创建并运行训练器
    trainer = trainer_cls(model, train_loader, test_loader, config)
    trainer.train()
    
    # 保存模型
    # 创建checkpoints/lenet目录（如果不存在）
    os.makedirs("checkpoints/lenet", exist_ok=True)
    # 保存模型到指定目录
    model_path = "checkpoints/lenet/lenet_mnist.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")
    
    # 预测流程
    # 从注册中心获取预测器
    predictor_cls = predictor_registry.get_predictor("mnist", "lenet_predictor")
    
    # 加载训练好的模型权重到新模型实例
    infer_model = model_cls()
    model_path = "checkpoints/lenet/lenet_mnist.pth"
    infer_model.load_state_dict(torch.load(model_path, map_location=config.device))
    
    # 实例化预测器
    predictor = predictor_cls(infer_model, config)
    
    # 从测试集中取一张图像作为示例
    test_dataset = test_loader.dataset
    sample_idx = 42  # 可修改为其他索引查看不同样本
    sample_image_tensor, sample_label = test_dataset[sample_idx]
    # 转换为PIL图像（预测器输入要求）
    pil_image = transforms.ToPILImage()(sample_image_tensor)
    
    # 执行预测
    pred_class, pred_confidence = predictor.predict(pil_image)
    
    # 输出预测结果
    print(f"\n示例预测结果：")
    print(f"真实标签：{sample_label}")
    print(f"预测标签：{pred_class}，置信度：{pred_confidence:.4f}")
    
    # 显示示例图像
    plt.imshow(pil_image, cmap='gray')
    plt.title(f"True: {sample_label}, Pred: {pred_class} (conf: {pred_confidence:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()