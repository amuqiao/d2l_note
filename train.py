import os
import sys
import argparse
from src.trainer.trainer import Trainer
from src.utils.model_registry import ModelRegistry
from src.utils.log_utils import get_logger


# 初始化日志，设置日志文件路径
logger = get_logger(
    name="train",
    log_file="logs/train.log",  # 日志文件路径，会自动创建logs目录
    global_level="DEBUG",     # 全局日志级别
    console_level="INFO",     # 控制台日志级别（只输出INFO及以上）
    file_level="DEBUG"        # 文件日志级别（输出所有级别）
)

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 导入模型并注册
from src.models.lenet import LeNet, LeNetBatchNorm
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet
from src.models.dense_net import DenseNet  # 使用装饰器自动注册的模型
from src.models.mlp import MLP  # 使用装饰器自动注册的模型


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度学习模型训练工具")
    
    # 获取已注册的模型列表，用于动态填充参数选项
    registered_models = ModelRegistry.list_models()
    
    # 训练参数
    parser.add_argument('--model_type', type=str, default='LeNet' if 'LeNet' in registered_models else registered_models[0] if registered_models else 'LeNet', 
                        choices=registered_models,
                        help=f"模型类型: {', '.join(registered_models)}")

    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练轮次（AlexNet建议10轮）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（AlexNet建议0.01）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--input_size', type=int, nargs=4, default=None,
                        help='输入尺寸，格式为：批量大小 通道数 高度 宽度（如：1 1 224 224）')
    parser.add_argument('--resize', type=int, default=None,
                        help='图像调整大小（LeNet不需要调整，默认为None）')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='是否每轮都保存模型文件（默认仅保存最佳模型）')
    parser.add_argument('--disable_visualization', action='store_true',
                        help='禁用实时可视化训练过程（默认启用）')
    
    parser.add_argument('--root_dir', type=str, default=None,
                        help='训练目录的根目录路径，默认为执行脚本的目录（当前工作目录）')
    
    parser.add_argument('--n', type=int, default=8,
                        help='训练后可视化样本数')
    
    return parser.parse_args()


def main():
    """训练脚本主函数"""
    # 解析命令行参数
    args = parse_arguments()
    logger.info(f"开始训练模型: {args.model_type}")
    
    # 创建训练器实例
    trainer = Trainer()
    
    # 获取模型配置
    config = trainer.get_model_config(
        model_type=args.model_type,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        input_size=args.input_size,
        resize=args.resize,
        root_dir=args.root_dir
    )
    
    # 执行训练
    enable_visualization = not args.disable_visualization
    try:
        result = trainer.run_training(
            model_type=args.model_type,
            config=config,
            enable_visualization=enable_visualization,
            save_every_epoch=args.save_every_epoch
        )
        
        # 训练后自动预测可视化
        trainer.run_post_training_prediction(
            run_dir=result["run_dir"],
            n=args.n,
            num_samples=10
        )
        
        logger.info(f"✅ 训练完成，最佳准确率: {result['best_accuracy']:.4f}")
        logger.info(f"📁 训练结果保存目录: {result['run_dir']}")
    except Exception as e:
        logger.error(f"❌ 训练过程出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()