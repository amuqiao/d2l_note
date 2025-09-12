import os
import sys
import argparse

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入增强版预测器
from src.predictor.predictor import Predictor


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度学习模型预测工具")
    
    # 预测模式参数
    parser.add_argument('--run_dir', type=str, default='data/run_20250909_204054',
                        help='训练目录路径（推荐方式）')
    parser.add_argument('--model_file', type=str, default=None,
                        help='可选，指定要加载的模型文件名，不指定则自动加载该目录下的最佳模型')
    parser.add_argument('--model_path', type=str, default='data/run_20250909_211745/best_model_LeNet_acc_0.8398_epoch_14.pth',
                        help='完整模型文件路径（最高优先级，设置后会忽略run_dir和model_file）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='预测时的批次大小（默认256）')
    parser.add_argument('--n', type=int, default=10,
                        help='可视化样本数')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='根目录路径，用于自动查找训练目录')
    
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
        
        print(f"✅ 预测完成")
        print(f"🔍 使用的模型: {result['model_name']}")
        if result['run_dir']:
            print(f"📁 训练目录: {result['run_dir']}")
        elif result['model_path']:
            print(f"📁 模型文件: {result['model_path']}")
    except Exception as e:
        print(f"❌ 预测过程出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()