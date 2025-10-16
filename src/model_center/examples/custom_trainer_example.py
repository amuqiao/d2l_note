import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
from src.model_center.trainer import BaseTrainer, ModelTrainerRegistry
from src.model_center.model_registry import ModelRegistry
from src.model_center.data_loader_registry import DataLoaderRegistry
from src.model_center.config_registry import ConfigRegistry
import logging

# 导入mnist_example以使用其中注册的模型和数据集
import src.model_center.examples.mnist_example

# 配置日志
logger = logging.getLogger(__name__)


@ModelTrainerRegistry.register(namespace="examples", name="mixed_precision")
class MixedPrecisionTrainer(BaseTrainer):
    """支持混合精度训练的自定义训练器
    
    这个训练器扩展了BaseTrainer，添加了混合精度训练功能，可以在不显著影响
    模型性能的情况下，减少内存使用并加快训练速度。
    """
    
    def __init__(self, config: dict):
        """初始化混合精度训练器
        
        Args:
            config: 训练配置字典，需包含mixed_precision配置项
        """
        super().__init__(config)
        
        # 初始化混合精度训练的scaler
        self.use_amp = config['training'].get('mixed_precision', {}).get('enabled', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        logger.info(f"混合精度训练已{'启用' if self.use_amp else '禁用'}")
    
    def train_one_epoch(self, epoch: int) -> tuple:
        """重写训练方法以支持混合精度训练
        
        Args:
            epoch: 当前epoch数
            
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
            
            # 混合精度训练上下文管理器
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 前向传播
                output = self.model(data)
                
                # 计算损失
                loss = self.loss_function(output, target)
            
            # 反向传播，使用scaler缩放梯度
            self.scaler.scale(loss).backward()
            
            # 更新权重
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
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
                            f"LR: {current_lr:.6f}\t" \
                            f"AMP: {'启用' if self.use_amp else '禁用'}")
        
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


@ModelTrainerRegistry.register(namespace="examples", name="gan_trainer")
class GANTrainer(BaseTrainer):
    """用于生成对抗网络(GAN)训练的自定义训练器
    
    这个训练器针对GAN的特殊训练流程进行了优化，支持单独训练生成器和判别器。
    """
    
    def __init__(self, config: dict):
        """初始化GAN训练器
        
        Args:
            config: 训练配置字典，需包含gan相关配置
        """
        super().__init__(config)
        
        # GAN特有的属性
        self.generator = None
        self.discriminator = None
        self.gen_optimizer = None
        self.disc_optimizer = None
        
        # 配置信息
        self.gan_config = config.get('gan', {})
        self.gen_steps = self.gan_config.get('generator_steps', 1)
        self.disc_steps = self.gan_config.get('discriminator_steps', 1)
        self.latent_dim = self.gan_config.get('latent_dim', 100)
    
    def _build_model(self) -> None:
        """重写模型构建方法，同时构建生成器和判别器"""
        # 构建生成器
        gen_config = self.config.get('generator', {})
        if gen_config and gen_config.get('name'):
            try:
                self.generator = ModelRegistry.create_model(
                    model_name=gen_config['name'],
                    namespace=gen_config.get('namespace', 'default'),
                    **gen_config.get('args', {})
                )
                self.generator.to(self.device)
                logger.info(f"已成功构建生成器: {gen_config['name']}")
            except Exception as e:
                logger.error(f"构建生成器 '{gen_config['name']}' 失败: {str(e)}")
                raise
        
        # 构建判别器
        disc_config = self.config.get('discriminator', {})
        if disc_config and disc_config.get('name'):
            try:
                self.discriminator = ModelRegistry.create_model(
                    model_name=disc_config['name'],
                    namespace=disc_config.get('namespace', 'default'),
                    **disc_config.get('args', {})
                )
                self.discriminator.to(self.device)
                logger.info(f"已成功构建判别器: {disc_config['name']}")
            except Exception as e:
                logger.error(f"构建判别器 '{disc_config['name']}' 失败: {str(e)}")
                raise
        
    def _build_optimizer(self) -> None:
        """重写优化器构建方法，为生成器和判别器分别创建优化器"""
        if not self.generator or not self.discriminator:
            logger.warning("生成器或判别器未初始化，跳过优化器构建")
            return
        
        # 生成器优化器
        gen_optim_config = self.gan_config.get('generator_optimizer', {'name': 'adam', 'args': {'lr': 0.0002}})
        gen_optim_name = gen_optim_config['name'].lower()
        gen_optim_args = gen_optim_config.get('args', {})
        
        if gen_optim_name == 'adam':
            self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), **gen_optim_args)
        elif gen_optim_name == 'sgd':
            self.gen_optimizer = torch.optim.SGD(self.generator.parameters(), **gen_optim_args)
        else:
            logger.error(f"不支持的生成器优化器: {gen_optim_name}")
            raise ValueError(f"不支持的生成器优化器: {gen_optim_name}")
        
        # 判别器优化器
        disc_optim_config = self.gan_config.get('discriminator_optimizer', {'name': 'adam', 'args': {'lr': 0.0002}})
        disc_optim_name = disc_optim_config['name'].lower()
        disc_optim_args = disc_optim_config.get('args', {})
        
        if disc_optim_name == 'adam':
            self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), **disc_optim_args)
        elif disc_optim_name == 'sgd':
            self.disc_optimizer = torch.optim.SGD(self.discriminator.parameters(), **disc_optim_args)
        else:
            logger.error(f"不支持的判别器优化器: {disc_optim_name}")
            raise ValueError(f"不支持的判别器优化器: {disc_optim_name}")
        
        logger.info(f"已成功构建GAN优化器")
    
    def build_components(self) -> None:
        """重写组件构建方法"""
        # 构建模型（生成器和判别器）
        self._build_model()
        
        # 构建优化器（生成器和判别器的优化器）
        self._build_optimizer()
        
        # 构建损失函数
        self._build_loss_function()
        
        # 构建数据加载器
        self._build_data_loaders()
    
    def train_one_epoch(self, epoch: int) -> tuple:
        """重写训练方法以支持GAN的特殊训练流程"""
        if not self.generator or not self.discriminator or not self.gen_optimizer or not self.disc_optimizer or not self.loss_function or not self.train_loader:
            raise RuntimeError("GAN训练组件未正确初始化")
        
        # 初始化指标
        gen_total_loss = 0.0
        disc_total_loss = 0.0
        
        start_time = time.time()
        
        # 训练判别器
        self.discriminator.train()
        for _ in range(self.disc_steps):
            for batch_idx, (real_images, _) in enumerate(self.train_loader):
                # 准备真实数据
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # 标签
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # 训练判别器
                self.disc_optimizer.zero_grad()
                
                # 判别真实图像
                real_output = self.discriminator(real_images)
                disc_real_loss = self.loss_function(real_output, real_labels)
                
                # 生成假图像并判别
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_images = self.generator(noise)
                fake_output = self.discriminator(fake_images.detach())
                disc_fake_loss = self.loss_function(fake_output, fake_labels)
                
                # 总损失
                disc_loss = disc_real_loss + disc_fake_loss
                disc_total_loss += disc_loss.item()
                
                # 反向传播和优化
                disc_loss.backward()
                self.disc_optimizer.step()
                
                # 打印训练进度
                if batch_idx % self.log_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} D [{batch_idx}/{len(self.train_loader)}]\t" \
                                f"Loss: {disc_loss.item():.6f}")
        
        # 训练生成器
        self.generator.train()
        for _ in range(self.gen_steps):
            for batch_idx, (_, _) in enumerate(self.train_loader):
                # 准备潜在向量
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                real_labels = torch.ones(batch_size, 1).to(self.device)
                
                # 训练生成器
                self.gen_optimizer.zero_grad()
                
                # 生成假图像
                fake_images = self.generator(noise)
                
                # 通过判别器
                outputs = self.discriminator(fake_images)
                
                # 计算损失（希望判别器认为假图像是真实的）
                gen_loss = self.loss_function(outputs, real_labels)
                gen_total_loss += gen_loss.item()
                
                # 反向传播和优化
                gen_loss.backward()
                self.gen_optimizer.step()
                
                # 打印训练进度
                if batch_idx % self.log_interval == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} G [{batch_idx}/{len(self.train_loader)}]\t" \
                                f"Loss: {gen_loss.item():.6f}")
        
        # 计算平均损失
        avg_gen_loss = gen_total_loss / (self.gen_steps * len(self.train_loader))
        avg_disc_loss = disc_total_loss / (self.disc_steps * len(self.train_loader))
        
        # 记录训练历史
        self.history['train_loss'].append(avg_gen_loss)
        self.history['discriminator_loss'] = self.history.get('discriminator_loss', [])
        self.history['discriminator_loss'].append(avg_disc_loss)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} 训练完成\t" \
                    f"生成器损失: {avg_gen_loss:.6f}\t" \
                    f"判别器损失: {avg_disc_loss:.6f}\t" \
                    f"耗时: {epoch_time:.2f}s")
        
        return avg_gen_loss, 0.0  # GAN没有传统意义上的准确率
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """重写保存检查点方法，保存生成器和判别器的状态"""
        if not self.generator or not self.discriminator:
            logger.warning("生成器或判别器未初始化，跳过保存检查点")
            return
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict() if self.gen_optimizer else None,
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict() if self.disc_optimizer else None,
            'gen_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'disc_loss': self.history['discriminator_loss'][-1] if 'discriminator_loss' in self.history else None,
            'config': self.config
        }
        
        # 保存当前检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"已保存GAN检查点: {checkpoint_path}")
        
        # 如果是最佳模型，保存为 best_model.pth
        if is_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"已保存最佳GAN模型: {best_checkpoint_path}")


# 导入必要的模块
import time
from src.model_center.config_registry import ConfigRegistry

# 注册配置模板，展示如何配置自定义训练器
ConfigRegistry.register_template("mixed_precision_mnist", {
    "model": {
        "name": "simple_cnn",
        "namespace": "mnist",
        "type": "cnn"
    },
    "data": {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
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
        }
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 10,
        "log_interval": 100,
        "checkpoint_dir": "./checkpoints/mixed_precision_mnist",
        "optimizer": {
            "name": "adam",
            "args": {
                "lr": 0.001
            }
        },
        "loss_function": {
            "name": "cross_entropy"
        },
        "mixed_precision": {
            "enabled": True
        }
    }
})


def main():
    """主函数：展示如何使用自定义训练器"""
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 加载配置
    config = ConfigRegistry.get_config("mixed_precision_mnist")
    
    # 使用训练器注册中心创建自定义训练器
    trainer = ModelTrainerRegistry.create_trainer(
        trainer_name="mixed_precision",
        config=config,
        namespace="examples"
    )
    
    # 开始训练
    trainer.train()
    
    # 示例：如何使用GAN训练器
    # gan_config = {...}  # 包含GAN训练所需的完整配置
    # gan_trainer = ModelTrainerRegistry.create_trainer(
    #     trainer_name="gan_trainer",
    #     config=gan_config,
    #     namespace="examples"
    # )
    # gan_trainer.train()


if __name__ == "__main__":
    main()