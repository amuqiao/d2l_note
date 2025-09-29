import os
import sys
import argparse
import os

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入增强版预测器
from src.predictor.predictor import Predictor

from src.utils.log_utils.log_utils import get_logger


# 初始化日志，设置日志文件路径
logger = get_logger(
    name=__name__,
    log_file=f"logs/predict.log",  # 日志文件路径，会自动创建logs目录
    global_level="DEBUG",     # 全局日志级别
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度学习模型预测工具")
    
    # 预测模式参数
    parser.add_argument('--run_dir', type=str, default=None,
                        help='训练目录路径（推荐方式）')
    parser.add_argument('--model_file', type=str, default=None,
                        help='可选，指定要加载的模型文件名，不指定则自动加载该目录下的最佳模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='完整模型文件路径（最高优先级，设置后会忽略run_dir和model_file）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='预测时的批次大小（默认256）')
    parser.add_argument('--n', type=int, default=10,
                        help='可视化样本数')
    parser.add_argument('--root_dir', type=str, default=os.path.abspath('.'),
                        help='根目录路径，用于自动查找训练目录（默认为当前目录）')
    
    return parser.parse_args()


def main():
    """预测脚本主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 执行预测
    try:
        # 直接创建增强版Predictor实例
        predictor = Predictor.create_from_args(
            run_dir=args.run_dir,
            model_file=args.model_file,
            model_path=args.model_path,
            root_dir=args.root_dir
        )
        
        # 执行预测
        result = predictor.run_prediction(
            batch_size=args.batch_size,
            num_samples=args.n
        )
        
        logger.info(f"✅ 预测完成")
        logger.info(f"🔍 使用的模型: {result['model_name']}")
        if result['run_dir']:
            logger.info(f"📁 训练目录: {result['run_dir']}")
        elif result['model_path']:
            logger.info(f"📁 模型文件: {result['model_path']}")
    except Exception as e:
        logger.exception(f"❌ 预测过程出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()