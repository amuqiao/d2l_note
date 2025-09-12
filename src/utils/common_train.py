import time
from src.utils.model_registry import ModelRegistry
from src.trainer.optimized_trainer import Trainer
from src.utils.data_utils import DataLoader


def train_model_common(model_name, config, enable_visualization=False, save_every_epoch=False):
    """
    统一的模型训练方法，可被run.py和batch_train.py共享
    
    Args:
        model_name: 模型名称
        config: 训练配置参数（包含num_epochs, lr, batch_size, resize等）
        enable_visualization: 是否启用可视化
        save_every_epoch: 是否每轮都保存模型
        
    Returns:
        Dict: 包含训练结果的字典
    """
    # 创建优化后的训练器实例
    trainer = Trainer()
    
    # 使用训练器的run_training方法执行训练
    result = trainer.run_training(
        model_type=model_name,
        config=config,
        enable_visualization=enable_visualization,
        save_every_epoch=save_every_epoch
    )
    
    return result