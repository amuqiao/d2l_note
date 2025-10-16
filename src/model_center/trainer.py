import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Type, Any, Callable, Tuple, Union
from src.model_center.model_registry import ModelRegistry
from src.model_center.data_loader_registry import DataLoaderRegistry
from src.model_center.config_registry import ConfigRegistry

# 配置日志
logger = logging.getLogger(__name__)


class BaseTrainer:
    """训练器基类：提供训练和评估的基础功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.train_loader = None
        self.test_loader = None
        self.device = torch.device(config['training'].get('device', 'cpu'))
        self.epochs = config['training'].get('epochs', 10)
        self.log_interval = config['training'].get('log_interval', 10)
        self.checkpoint_dir = config['training'].get('checkpoint_dir', './checkpoints')
        self.early_stopping = config['training'].get('early_stopping', {'enabled': False, 'patience': 5})
        
        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化训练状态
        self.best_score = None
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': []
        }
    
    def build_components(self) -> None:
        """构建训练所需的组件：模型、优化器、损失函数和数据加载器"""
        # 1. 构建模型
        self._build_model()
        
        # 2. 构建优化器
        self._build_optimizer()
        
        # 3. 构建学习率调度器
        self._build_scheduler()
        
        # 4. 构建损失函数
        self._build_loss_function()
        
        # 5. 构建数据加载器
        self._build_data_loaders()
        
        # 6. 将模型移至指定设备
        if self.model:
            self.model.to(self.device)
            logger.info(f"模型已移至设备: {self.device}")
    
    def _build_model(self) -> None:
        """构建模型"""
        model_config = self.config.get('model', {})
        if not model_config or not model_config.get('name'):
            logger.warning("未配置模型，跳过模型构建")
            return
        
        model_name = model_config['name']
        model_args = model_config.get('args', {})
        model_namespace = model_config.get('namespace', 'default')
        model_type = model_config.get('type', 'default')
        
        try:
            # 使用模型注册中心创建模型实例
            self.model = ModelRegistry.create_model(
                model_name=model_name,
                namespace=model_namespace,
                type=model_type,
                **model_args
            )
            logger.info(f"已成功构建模型: {model_name}")
        except Exception as e:
            logger.error(f"构建模型 '{model_name}' 失败: {str(e)}")
            raise
    
    def _build_optimizer(self) -> None:
        """构建优化器"""
        if not self.model:
            logger.warning("模型未初始化，跳过优化器构建")
            return
        
        optimizer_config = self.config['training'].get('optimizer', {'name': 'adam', 'args': {'lr': 0.001}})
        optimizer_name = optimizer_config['name'].lower()
        optimizer_args = optimizer_config.get('args', {})
        
        # 获取模型参数
        model_params = self.model.parameters()
        
        # 根据名称创建优化器
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model_params, **optimizer_args)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model_params, **optimizer_args)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(model_params, **optimizer_args)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(model_params, **optimizer_args)
        else:
            logger.error(f"不支持的优化器: {optimizer_name}")
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        logger.info(f"已成功构建优化器: {optimizer_name}")
    
    def _build_scheduler(self) -> None:
        """构建学习率调度器"""
        if not self.optimizer:
            logger.warning("优化器未初始化，跳过学习率调度器构建")
            return
        
        scheduler_config = self.config['training'].get('scheduler', {})
        if not scheduler_config or not scheduler_config.get('name'):
            logger.info("未配置学习率调度器")
            return
        
        scheduler_name = scheduler_config['name'].lower()
        scheduler_args = scheduler_config.get('args', {})
        
        # 根据名称创建学习率调度器
        if scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_args)
        elif scheduler_name == 'multi_step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **scheduler_args)
        elif scheduler_name == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_args)
        elif scheduler_name == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_args)
        elif scheduler_name == 'reduce_lr_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_args)
        else:
            logger.error(f"不支持的学习率调度器: {scheduler_name}")
            raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
        
        logger.info(f"已成功构建学习率调度器: {scheduler_name}")
    
    def _build_loss_function(self) -> None:
        """构建损失函数"""
        loss_config = self.config['training'].get('loss_function', {'name': 'cross_entropy', 'args': {}})
        loss_name = loss_config['name'].lower()
        loss_args = loss_config.get('args', {})
        
        # 根据名称创建损失函数
        if loss_name == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss(**loss_args)
        elif loss_name == 'mse':
            self.loss_function = nn.MSELoss(**loss_args)
        elif loss_name == 'bce':
            self.loss_function = nn.BCELoss(**loss_args)
        elif loss_name == 'bce_with_logits':
            self.loss_function = nn.BCEWithLogitsLoss(**loss_args)
        elif loss_name == 'l1':
            self.loss_function = nn.L1Loss(**loss_args)
        else:
            logger.error(f"不支持的损失函数: {loss_name}")
            raise ValueError(f"不支持的损失函数: {loss_name}")
        
        logger.info(f"已成功构建损失函数: {loss_name}")
    
    def _build_data_loaders(self) -> None:
        """构建训练和测试数据加载器"""
        data_config = self.config.get('data', {})
        if not data_config:
            logger.warning("未配置数据，跳过数据加载器构建")
            return
        
        # 获取通用数据配置
        batch_size = data_config.get('batch_size', 32)
        shuffle = data_config.get('shuffle', True)
        num_workers = data_config.get('num_workers', 0)
        
        # 构建训练数据加载器
        train_dataset_config = data_config.get('train_dataset', {})
        if train_dataset_config and train_dataset_config.get('name'):
            try:
                self.train_loader = DataLoaderRegistry.create_data_loader(
                    dataset_name=train_dataset_config['name'],
                    namespace=train_dataset_config.get('namespace', 'default'),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    **train_dataset_config.get('args', {})
                )
                logger.info(f"已成功构建训练数据加载器: {train_dataset_config['name']}")
            except Exception as e:
                logger.error(f"构建训练数据加载器失败: {str(e)}")
                raise
        
        # 构建测试数据加载器
        test_dataset_config = data_config.get('test_dataset', {})
        if test_dataset_config and test_dataset_config.get('name'):
            try:
                self.test_loader = DataLoaderRegistry.create_data_loader(
                    dataset_name=test_dataset_config['name'],
                    namespace=test_dataset_config.get('namespace', 'default'),
                    batch_size=batch_size,
                    shuffle=False,  # 测试数据通常不打乱
                    num_workers=num_workers,
                    **test_dataset_config.get('args', {})
                )
                logger.info(f"已成功构建测试数据加载器: {test_dataset_config['name']}")
            except Exception as e:
                logger.error(f"构建测试数据加载器失败: {str(e)}")
                raise
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个 epoch
        
        Args:
            epoch: 当前 epoch 数
            
        Returns:
            (平均训练损失, 平均训练准确率)
        """
        if not self.model or not self.optimizer or not self.loss_function or not self.train_loader:
            raise RuntimeError("训练组件未正确初始化")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 将数据移至指定设备
            data, target = data.to(self.device), target.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失
            loss = self.loss_function(output, target)
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            
            # 计算准确率（如果是分类任务）
            if len(output.shape) == 2 and output.shape[1] > 1:  # 多分类
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # 打印训练进度
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{self.epochs} [{batch_idx}/{len(self.train_loader)}]\t" \
                            f"Loss: {loss.item():.6f}\t" \
                            f"LR: {current_lr:.6f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        # 计算准确率
        if total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = 0.0
        
        # 记录训练历史
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} 训练完成\t" \
                    f"平均损失: {avg_loss:.6f}\t" \
                    f"准确率: {accuracy:.2f}%\t" \
                    f"耗时: {epoch_time:.2f}s")
        
        return avg_loss, accuracy
    
    def evaluate(self) -> Tuple[float, float]:
        """评估模型性能
        
        Returns:
            (平均评估损失, 平均评估准确率)
        """
        if not self.model or not self.loss_function or not self.test_loader:
            raise RuntimeError("评估组件未正确初始化")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                # 将数据移至指定设备
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失
                total_loss += self.loss_function(output, target).item()
                
                # 计算准确率（如果是分类任务）
                if len(output.shape) == 2 and output.shape[1] > 1:  # 多分类
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.test_loader)
        
        # 计算准确率
        if total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = 0.0
        
        # 记录评估历史
        self.history['test_loss'].append(avg_loss)
        self.history['test_acc'].append(accuracy)
        
        logger.info(f"评估完成\t平均损失: {avg_loss:.6f}\t准确率: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def train(self) -> None:
        """执行完整的训练流程"""
        logger.info("开始训练流程")
        
        # 确保所有组件都已构建
        if not self.model or not self.optimizer or not self.loss_function or not self.train_loader:
            logger.error("训练组件未正确初始化，无法开始训练")
            raise RuntimeError("训练组件未正确初始化，无法开始训练")
        
        # 训练循环
        for epoch in range(self.epochs):
            logger.info(f"===== Epoch {epoch+1}/{self.epochs} =====")
            
            # 训练一个 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # 评估模型
            if self.test_loader:
                test_loss, test_acc = self.evaluate()
            else:
                logger.warning("未配置测试数据加载器，跳过评估")
                test_acc = 0.0
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_loss)
                else:
                    self.scheduler.step()
            
            # 检查是否是最佳模型
            if self.test_loader:
                score = test_acc  # 使用测试准确率作为评分标准
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch + 1
                    self.early_stopping_counter = 0
                    # 保存最佳模型
                    self.save_checkpoint(epoch + 1, is_best=True)
                else:
                    self.early_stopping_counter += 1
            
            # 保存当前模型
            self.save_checkpoint(epoch + 1)
            
            # 检查早停条件
            if self.early_stopping['enabled'] and self.early_stopping_counter >= self.early_stopping['patience']:
                logger.info(f"早停触发，停止训练。最佳模型在第 {self.best_epoch} 轮，准确率为 {self.best_score:.2f}%")
                break
        
        logger.info(f"训练完成！最佳模型在第 {self.best_epoch} 轮，准确率为 {self.best_score:.2f}%")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型检查点
        
        Args:
            epoch: 当前 epoch 数
            is_best: 是否是最佳模型
        """
        if not self.model or not self.optimizer:
            logger.warning("模型或优化器未初始化，跳过保存检查点")
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'accuracy': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'config': self.config
        }
        
        # 如果有学习率调度器，也保存其状态
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存当前检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"已保存检查点: {checkpoint_path}")
        
        # 如果是最佳模型，保存为 best_model.pth
        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"已保存最佳模型: {best_checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        if self.model:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("已加载模型状态")
        
        # 加载优化器状态
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("已加载优化器状态")
        
        # 加载学习率调度器状态
        if self.scheduler and 'scheduler_state_dict' in checkpoint and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("已加载学习率调度器状态")
        
        # 更新配置
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            logger.info("已加载检查点配置")
        
        logger.info(f"已成功加载检查点: {checkpoint_path}")


class ModelTrainerRegistry:
    """模型训练器注册中心：用于管理和创建不同类型的训练器"""
    
    # 存储训练器类的字典，格式：{namespace: {trainer_name: trainer_class}}
    _trainers: Dict[str, Dict[str, Type[BaseTrainer]]] = {"default": {}}
    
    @classmethod
    def register(cls, trainer_class=None, namespace: str = "default", name: Optional[str] = None) -> Type[BaseTrainer] or Callable:
        """注册训练器类（支持直接调用和装饰器用法）
        
        Args:
            trainer_class: 训练器类
            namespace: 命名空间，默认为"default"
            name: 训练器名称，如果不提供则使用类名
            
        Returns:
            训练器类（用于直接调用）或装饰器函数（用于带参数的装饰器用法）
        """
        # 如果trainer_class为None，说明是作为带参数的装饰器使用
        if trainer_class is None:
            # 返回一个可以接受trainer_class的装饰器函数
            def decorator(cls_to_register):
                return cls.register(cls_to_register, namespace, name)
            return decorator
        
        # 确定训练器名称
        trainer_name = name if name else trainer_class.__name__
        
        # 确保命名空间存在
        if namespace not in cls._trainers:
            cls._trainers[namespace] = {}
        
        # 注册训练器类
        cls._trainers[namespace][trainer_name] = trainer_class
        logger.debug(f"训练器 '{trainer_name}' 已在命名空间 '{namespace}' 中注册")
        
        return trainer_class
    
    @classmethod
    def get(cls, trainer_name: str, namespace: str = "default") -> Optional[Type[BaseTrainer]]:
        """获取训练器类
        
        Args:
            trainer_name: 训练器名称
            namespace: 命名空间，默认为"default"
            
        Returns:
            训练器类，如果未找到则返回None
        """
        # 首先在指定命名空间中查找
        if namespace in cls._trainers and trainer_name in cls._trainers[namespace]:
            return cls._trainers[namespace][trainer_name]
        
        # 如果指定命名空间中未找到，尝试在默认命名空间中查找
        if namespace != "default" and "default" in cls._trainers and trainer_name in cls._trainers["default"]:
            logger.debug(f"在命名空间 '{namespace}' 中未找到训练器 '{trainer_name}'，使用默认命名空间中的训练器")
            return cls._trainers["default"][trainer_name]
        
        logger.warning(f"未找到训练器 '{trainer_name}'（命名空间: '{namespace}'）")
        return None
    
    @classmethod
    def create_trainer(cls, trainer_name: str, config: Dict[str, Any], 
                      namespace: str = "default", **kwargs) -> BaseTrainer:
        """创建训练器实例
        
        Args:
            trainer_name: 训练器名称
            config: 训练配置
            namespace: 命名空间，默认为"default"
            **kwargs: 传递给训练器构造函数的额外参数
            
        Returns:
            训练器实例
        """
        trainer_class = cls.get(trainer_name, namespace)
        if not trainer_class:
            registered_trainers = ', '.join(cls.list_trainers(namespace))
            raise ValueError(
                f"不支持的训练器类型: {trainer_name}，命名空间 '{namespace}' 中已注册的训练器有: {registered_trainers}"
            )
        
        # 创建训练器实例
        trainer = trainer_class(config=config, **kwargs)
        logger.info(f"已创建训练器实例: {trainer_name}")
        
        # 构建训练器组件
        trainer.build_components()
        
        return trainer
    
    @classmethod
    def list_trainers(cls, namespace: str = "default") -> List[str]:
        """列出指定命名空间中所有已注册的训练器名称
        
        Args:
            namespace: 命名空间，默认为"default"
            
        Returns:
            训练器名称列表
        """
        if namespace in cls._trainers:
            return list(cls._trainers[namespace].keys())
        return []


# 注册默认训练器
ModelTrainerRegistry.register(BaseTrainer, name="default")


def create_default_trainer(config: Dict[str, Any]) -> BaseTrainer:
    """创建默认训练器实例
    
    Args:
        config: 训练配置
        
    Returns:
        训练器实例
    """
    trainer = ModelTrainerRegistry.create_trainer("default", config)
    return trainer


def train_model(config: Dict[str, Any]) -> BaseTrainer:
    """训练模型的便捷函数
    
    Args:
        config: 训练配置
        
    Returns:
        训练器实例
    """
    # 创建训练器
    trainer = create_default_trainer(config)
    
    # 开始训练
    trainer.train()
    
    return trainer