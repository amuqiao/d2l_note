import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
import json
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List

# 导入基类和注册中心
from light_base_trainer import LightBaseTrainerTemplate
from registry_center import DataLoaderRegistry, ModelRegistry, ConfigRegistry, TrainerRegistry, PredictorRegistry

# 设置环境变量解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 初始化注册中心实例
data_loader_registry = DataLoaderRegistry()
model_registry = ModelRegistry()
config_registry = ConfigRegistry()
trainer_registry = TrainerRegistry()
predictor_registry = PredictorRegistry()

# 配置项
@config_registry.register_config("flower_classification", "base_config")
class FlowerConfig:
    def __init__(self):
        # 数据集路径
        self.data_dir = "./data/flower_data"
        
        # 类别数量
        self.num_classes = 102
        # 批量大小 - 增加到64以更好地利用GPU内存并提高训练稳定性
        self.batch_size = 64
        # 学习率 - 为特征提取设置合适的学习率
        self.lr = 0.001  # 特征提取的学习率
        self.fine_tuning_lr = 0.0001  # 微调的学习率
        # 训练轮数 - 增加到30以获得更好的模型性能
        self.num_epochs = 30
        # 设备选择
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 预训练模型名称
        self.model_name = "resnet18"
        # 模型类型 (feature_extraction 或 fine_tuning)
        self.model_type = "fine_tuning"  # 默认为微调以获得更好的性能
        # 解冻层数（仅用于fine_tuning）
        self.unfreeze_layers = 2  # 解冻最后两层卷积层
        # 学习率调度器参数
        self.scheduler_step_size = 10  # 每10个epoch调整一次学习率
        self.scheduler_gamma = 0.1  # 学习率衰减因子

# 数据加载器
@data_loader_registry.register_data_loader("flower_classification", "flower_dataloader")
class FlowerDataLoader:
    def __init__(self, config: FlowerConfig):
        self.config = config
        self._prepare_data_transforms()
        self._create_datasets()
        self._create_dataloaders()
    
    def _prepare_data_transforms(self):
        # 训练数据增强和标准化
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 验证数据标准化
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _create_datasets(self):
        # 创建训练集和验证集
        self.image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.config.data_dir, 'train'), self.train_transforms),
            'val': datasets.ImageFolder(os.path.join(self.config.data_dir, 'val'), self.val_transforms)
        }
        
        # 类别名称
        self.class_names = self.image_datasets['train'].classes
    
    def _create_dataloaders(self):
        # 创建数据加载器
        self.dataloaders = {
            'train': DataLoader(self.image_datasets['train'], batch_size=self.config.batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(self.image_datasets['val'], batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        }
        
        # 数据集大小
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

# 迁移学习模型 - 特征提取方式（冻结基础层）
@model_registry.register_model("flower_classification", "feature_extraction", "resnet18")
class FeatureExtractionResNet18:
    def __init__(self, config: FlowerConfig):
        self.config = config
        self._initialize_model()
    
    def _initialize_model(self):
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(pretrained=True)
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 获取最后一个全连接层的输入特征数
        num_ftrs = self.model.fc.in_features
        
        # 替换最后一个全连接层，用于102类分类
        self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        
        # 将模型移至指定设备
        self.model = self.model.to(self.config.device)

# 迁移学习模型 - 微调方式（解冻部分层）
@model_registry.register_model("flower_classification", "fine_tuning", "resnet18")
class FineTuningResNet18:
    def __init__(self, config: FlowerConfig):
        self.config = config
        self._initialize_model()
    
    def _initialize_model(self):
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(pretrained=True)
        
        # 默认冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻最后几个卷积层（根据unfreeze_layers参数）
        if self.config.unfreeze_layers >= 1:
            # 解冻layer4
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        
        if self.config.unfreeze_layers >= 2:
            # 解冻layer3
            for param in self.model.layer3.parameters():
                param.requires_grad = True
        
        # 获取最后一个全连接层的输入特征数
        num_ftrs = self.model.fc.in_features
        
        # 替换最后一个全连接层，用于102类分类
        self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        
        # 将模型移至指定设备
        self.model = self.model.to(self.config.device)

# 训练器 - 继承自LightBaseTrainerTemplate
@trainer_registry.register_trainer("flower_classification", "transfer_learning_trainer")
class TransferLearningTrainer(LightBaseTrainerTemplate):
    def __init__(self, config: FlowerConfig, save_dir: str = "./checkpoints"):
        super().__init__(config, save_dir)
        self.dataloader = None
        self.class_names = None
        self.metrics = []  # 用于存储所有epoch的指标
        self.current_epoch_metrics = {}  # 用于存储当前epoch的指标
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        # 构建所有组件
        self.build_all_components()
        # 保存配置文件
        self._save_config()

    def _save_config(self):
        """保存配置到config.json文件"""
        # 将配置转换为字典
        config_dict = vars(self.config).copy()
        # 确保device对象能被JSON序列化
        if hasattr(config_dict.get('device', None), 'type'):
            config_dict['device'] = str(config_dict['device'])
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
    
    # -------------------------- 核心组件构建接口实现 --------------------------
    def _build_model(self) -> None:
        """构建模型并赋值给self.model"""
        if self.config.model_type == "feature_extraction":
            model_wrapper = model_registry.get_model(
                "flower_classification", "feature_extraction", "resnet18")(self.config)
        else:  # fine_tuning
            model_wrapper = model_registry.get_model(
                "flower_classification", "fine_tuning", "resnet18")(self.config)
        self.model = model_wrapper.model
    
    def _build_optimizer(self) -> None:
        """构建优化器并赋值给self.optimizer"""
        # 只对requires_grad=True的参数进行优化
        params_to_update = []
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        
        # 根据模型类型选择合适的学习率
        if self.config.model_type == "fine_tuning":
            learning_rate = self.config.fine_tuning_lr
        else:
            learning_rate = self.config.lr
        
        self.optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
    
    def _build_scheduler(self) -> None:
        """构建学习率调度器"""
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config.scheduler_step_size, 
            gamma=self.config.scheduler_gamma
        )
    
    def _build_loss_function(self) -> None:
        """构建损失函数"""
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _build_data_loaders(self) -> None:
        """构建数据加载器"""
        self.dataloader = data_loader_registry.get_data_loader(
            "flower_classification", "flower_dataloader")(self.config)
        self.train_loader = self.dataloader.dataloaders['train']
        self.test_loader = self.dataloader.dataloaders['val']
        self.class_names = self.dataloader.class_names
    
    def _init_metrics(self) -> None:
        """初始化指标规范"""
        self.best_metric_name = "eval_acc"
        self.best_metric = 0.0  # 准确率越大越好，初始值设为0
    
    def set_model_mode(self, mode: str) -> None:
        """设置模型模式（train/eval）"""
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    # -------------------------- 核心流程接口实现 --------------------------
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """单轮训练逻辑"""
        self.set_model_mode("train")
        
        running_loss = 0.0
        running_corrects = 0
        
        # 迭代数据
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_fn(outputs, labels)
                
                # 反向传播 + 优化
                loss.backward()
                self.optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 更新学习率调度器
        self.scheduler.step()
        
        # 计算平均损失和准确率
        epoch_loss = running_loss / self.dataloader.dataset_sizes['train']
        epoch_acc = running_corrects.double() / self.dataloader.dataset_sizes['train']
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        return {"loss": epoch_loss, "acc": epoch_acc.item()}
    
    def evaluate(self) -> Dict[str, float]:
        """模型评估逻辑"""
        self.set_model_mode("eval")
        
        running_loss = 0.0
        running_corrects = 0
        
        # 迭代数据
        for inputs, labels in self.test_loader:
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播（不计算梯度）
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.loss_fn(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # 计算平均损失和准确率
        epoch_loss = running_loss / self.dataloader.dataset_sizes['val']
        epoch_acc = running_corrects.double() / self.dataloader.dataset_sizes['val']
        
        print(f'Eval Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        return {"loss": epoch_loss, "acc": epoch_acc.item()}
    
    def train(self) -> None:
        """完整训练流程"""
        since = time.time()
        
        for self.epoch in range(self.config.num_epochs):
            print(f'Epoch {self.epoch}/{self.config.num_epochs - 1}')
            print('-' * 10)
            
            # 训练一轮
            train_metrics = self.train_one_epoch(self.epoch)
            
            # 评估
            eval_metrics = self.evaluate()
            
            # 计算耗时
            epoch_time = time.time() - since
            since = time.time()
            
            # 记录指标并判断是否为最佳模型
            is_best = self.record_epoch_metrics(self.epoch, train_metrics, eval_metrics, epoch_time)
            
            # 保存checkpoint
            self.save_checkpoint(self.epoch, is_best)
            
            print()
        
        print(f'Training complete. Best {self.best_metric_name}: {self.best_metric:4f} at epoch {self.best_epoch}')
    
    def predict(self, input_data: Any, device: Optional[str] = None) -> Tuple[Any, ...]:
        """模型推理逻辑"""
        self.set_model_mode("eval")
        
        # 处理设备
        if device is None:
            device = self.config.device
        
        # 预处理图像
        if not isinstance(input_data, torch.Tensor):
            input_data = self.dataloader.val_transforms(input_data).unsqueeze(0)
        
        input_data = input_data.to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(input_data)
            _, preds = torch.max(outputs, 1)
        
        # 计算置信度
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]].item()
        class_name = self.class_names[preds[0]]
        
        return class_name, confidence, preds[0].item(), confidence
    
    # -------------------------- 文件与指标处理接口实现 --------------------------
    def record_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                           eval_metrics: Dict[str, float], epoch_time: float) -> bool:
        """记录当前轮次指标"""
        # 构建当前轮次指标字典
        # 确保包含 LightBaseTrainerTemplate 要求的所有 key
        self.current_epoch_metrics = {
            "epoch": epoch,
            "epoch_time": epoch_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train": {
                "loss": train_metrics["loss"],
                "acc": train_metrics["acc"]
            },
            "eval": {
                "loss": eval_metrics["loss"],
                "acc": eval_metrics["acc"]
            },
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "best_epoch": self.best_epoch
        }
        
        # 添加到历史指标列表
        self.metrics.append(self.current_epoch_metrics)
        
        # 检查是否为最佳模型
        current_metric = eval_metrics["acc"]
        is_best = current_metric > self.best_metric
        
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
            # 更新当前轮次指标中的最佳值
            self.current_epoch_metrics["best_metric"] = self.best_metric
            self.current_epoch_metrics["best_epoch"] = self.best_epoch
        
        # 保存指标到文件
        self._save_metrics()
        
        return is_best
    
    def _save_metrics(self):
        """保存指标到json文件"""
        # 保存所有轮次指标到metrics.json
        # 包含epoch列表和最终训练指标
        metrics_data = {
            "epochs": self.metrics,
            "final_metrics": {
                "best_epoch": self.best_epoch,
                "best_metric_name": self.best_metric_name,
                "best_metric_value": self.best_metric,
                "total_epochs": self.config.num_epochs
            }
        }
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics_data, f, indent=4)
        
        # 保存当前轮次指标到epoch_metrics.json（JSON列表格式）
        # 检查文件是否存在，如果存在则读取现有内容，否则创建空列表
        epoch_metrics_path = os.path.join(self.save_dir, "epoch_metrics.json")
        if os.path.exists(epoch_metrics_path):
            try:
                with open(epoch_metrics_path, "r") as f:
                    epoch_metrics_list = json.load(f)
            except json.JSONDecodeError:
                # 如果文件存在但格式错误，创建空列表
                epoch_metrics_list = []
        else:
            epoch_metrics_list = []
        
        # 添加当前轮次指标到列表
        epoch_metrics_list.append(self.current_epoch_metrics)
        
        # 保存更新后的列表
        with open(epoch_metrics_path, "w") as f:
            json.dump(epoch_metrics_list, f, indent=4)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "config": vars(self.config),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None
        }
        
        # 保存当前轮次checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"已保存checkpoint到 {checkpoint_path}")
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"已保存最佳模型到 {best_checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载模型checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint["model_state"])
        
        # 加载优化器状态
        if "optimizer_state" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # 加载调度器状态
        if "scheduler_state" in checkpoint and self.scheduler and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        # 恢复训练状态
        self.epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.best_metric_name = checkpoint["best_metric_name"]
        
        print(f"已加载checkpoint: {checkpoint_path}, 最佳{self.best_metric_name}: {self.best_metric}")
        return checkpoint

# 预测器
@predictor_registry.register_predictor("flower_classification", "flower_predictor")
class FlowerPredictor:
    def __init__(self, trainer: TransferLearningTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataloader = trainer.dataloader
        self.config = trainer.config
    
    def predict(self, image):
        # 使用训练器的predict方法
        class_name, confidence, _, _ = self.trainer.predict(image)
        return class_name, confidence

# 下载Flower数据集（保持不变）
def download_flower_data(data_dir: str = "./flower_data"):
    import requests
    import zipfile
    import io
    import scipy.io
    import shutil
    import os
    from tqdm import tqdm
    
    # 数据集下载链接
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    setid_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 增强数据集检测逻辑，避免重复下载
    def is_dataset_complete():
        # 检查必要的目录是否存在
        required_dirs = ['train', 'val', 'test']
        for dir_name in required_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                return False
            
            # 检查每个目录是否至少有一个类别的文件夹
            if len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]) == 0:
                return False
        
        # 检查是否有图像文件
        sample_img_path = os.path.join(data_dir, 'train', '1', 'image_00001.jpg')
        if not os.path.exists(sample_img_path):
            # 尝试查找任意一个图像文件
            has_images = False
            for root, _, files in os.walk(os.path.join(data_dir, 'train')):
                if any(f.endswith('.jpg') for f in files):
                    has_images = True
                    break
            if not has_images:
                return False
        
        return True
    
    # 检查数据集是否已经完整存在
    if is_dataset_complete():
        print(f"数据集已经存在且完整于 {data_dir}，跳过下载。")
        return
    
    # 检查是否已有部分数据集文件
    if (os.path.exists(os.path.join(data_dir, '102flowers.tgz')) or 
        os.path.exists(os.path.join(data_dir, 'imagelabels.mat')) or 
        os.path.exists(os.path.join(data_dir, 'setid.mat')) or 
        os.path.exists(os.path.join(data_dir, 'jpg'))):
        print(f"检测到 {data_dir} 中有部分数据集文件，将尝试继续处理...")
    else:
        print("开始下载Flower数据集...")
    
    try:
        # 下载并解压花卉图像
        if not os.path.exists(os.path.join(data_dir, 'jpg')):
            if not os.path.exists(os.path.join(data_dir, '102flowers.tgz')):
                print("1. 下载花卉图像...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB chunks
                
                with open(os.path.join(data_dir, '102flowers.tgz'), 'wb') as f:
                    for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='MB'):
                        f.write(data)
            
            print("解压花卉图像...")
            import tarfile
            with tarfile.open(os.path.join(data_dir, '102flowers.tgz'), 'r:gz') as tar:
                tar.extractall(path=data_dir)
        else:
            print("花卉图像目录已存在，跳过下载和解压。")
        
        # 下载标签文件
        if not os.path.exists(os.path.join(data_dir, 'imagelabels.mat')):
            print("2. 下载标签文件...")
            labels_response = requests.get(labels_url)
            labels_response.raise_for_status()
            with open(os.path.join(data_dir, 'imagelabels.mat'), 'wb') as f:
                f.write(labels_response.content)
        else:
            print("标签文件已存在，跳过下载。")
        
        # 下载数据集划分文件
        if not os.path.exists(os.path.join(data_dir, 'setid.mat')):
            print("3. 下载数据集划分文件...")
            setid_response = requests.get(setid_url)
            setid_response.raise_for_status()
            with open(os.path.join(data_dir, 'setid.mat'), 'wb') as f:
                f.write(setid_response.content)
        else:
            print("数据集划分文件已存在，跳过下载。")
        
        # 处理数据集
        if not all(os.path.exists(os.path.join(data_dir, dir_name)) for dir_name in ['train', 'val', 'test']):
            print("处理数据集并创建训练/验证/测试目录...")
            
            # 读取标签和数据集划分
            labels = scipy.io.loadmat(os.path.join(data_dir, 'imagelabels.mat'))['labels'][0]
            train_ids = scipy.io.loadmat(os.path.join(data_dir, 'setid.mat'))['trnid'][0]
            val_ids = scipy.io.loadmat(os.path.join(data_dir, 'setid.mat'))['valid'][0]
            test_ids = scipy.io.loadmat(os.path.join(data_dir, 'setid.mat'))['tstid'][0]
            
            # 创建训练、验证、测试目录
            splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
            
            for split, ids in splits.items():
                # 仅当目录不存在时才创建和处理
                if not os.path.exists(os.path.join(data_dir, split)) or len(os.listdir(os.path.join(data_dir, split))) == 0:
                    for img_id in ids:
                        # 图像文件名格式为 'image_xxxxx.jpg'，其中xxxxx是5位数字
                        img_name = f'image_{img_id:05d}.jpg'
                        src_path = os.path.join(data_dir, 'jpg', img_name)
                        
                        # 图像标签（从1开始）
                        label = labels[img_id - 1]
                        dst_dir = os.path.join(data_dir, split, str(label))
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        # 复制图像到对应目录
                        if os.path.exists(src_path):
                            shutil.copy(src_path, dst_dir)
        else:
            print("训练/验证/测试目录已存在，跳过处理。")
        
        # 最后验证数据集是否完整
        if is_dataset_complete():
            print("数据集准备完成!")
        else:
            print("警告：数据集可能不完整，请检查目录结构和文件。")
        
    except Exception as e:
        print(f"下载或处理数据集时出错: {e}")
        print("\n如果自动下载失败，请手动下载以下文件并按照README中的说明处理:")
        print(f"1. 花卉图像: {url}")
        print(f"2. 标签文件: {labels_url}")
        print(f"3. 数据集划分文件: {setid_url}")
    print("\n然后按照代码中的数据加载器要求组织数据结构。")

# 主函数
def main():
    # 下载数据集
    config = FlowerConfig()
    download_flower_data(config.data_dir)
    
    # 检查数据集是否存在且完整
    def check_dataset_integrity(data_dir):
        # 检查必要的目录是否存在
        required_dirs = ['train', 'val']
        for dir_name in required_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                print(f"数据集目录不完整：缺少{dir_name}目录")
                return False
            
            # 检查每个目录是否至少有一个类别的文件夹
            if len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]) == 0:
                print(f"数据集目录不完整：{dir_name}目录为空")
                return False
        
        # 检查是否有图像文件
        has_images = False
        for root, _, files in os.walk(os.path.join(data_dir, 'train')):
            if any(f.endswith('.jpg') for f in files):
                has_images = True
                break
        if not has_images:
            print("数据集不完整：未找到图像文件")
            return False
        
        return True
    
    # 检查数据集是否存在且完整
    if not os.path.exists(config.data_dir):
        print(f"数据集目录 {config.data_dir} 不存在，请先下载数据集。")
        return
    
    if not check_dataset_integrity(config.data_dir):
        print("请重新运行脚本以下载完整的数据集，或手动检查数据集完整性。")
        return
    
    print("\n=== 方法1: 特征提取（冻结基础层）===")
    try:
        # 创建特征提取配置
        fe_config = FlowerConfig()
        fe_config.model_type = "feature_extraction"
        
        # 创建训练器
        fe_trainer = TransferLearningTrainer(fe_config, save_dir="./checkpoints/feature_extraction")
        
        # 训练模型
        print("开始训练特征提取模型...")
        fe_trainer.train()
        print("特征提取模型训练完成")
    except Exception as e:
        print(f"特征提取模型训练出错: {e}")
    
    print("\n=== 方法2: 微调（解冻部分层）===")
    try:
        # 创建微调配置
        ft_config = FlowerConfig()
        ft_config.model_type = "fine_tuning"
        ft_config.unfreeze_layers = 2
        
        # 创建训练器
        ft_trainer = TransferLearningTrainer(ft_config, save_dir="./checkpoints/fine_tuning")
        
        # 训练模型
        print("开始训练微调模型...")
        ft_trainer.train()
        print("微调模型训练完成")
    except Exception as e:
        print(f"微调模型训练出错: {e}")

if __name__ == "__main__":
    main()