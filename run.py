import os
import sys
import argparse
import subprocess
import os

# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入日志工具
from src.utils.log_utils import init_logger, info, error


# ========================= 主函数入口 =========================
def main():
    """
    主函数：作为训练和预测功能的统一入口
    负责解析命令行参数并调用相应的专用脚本
    """
    # 初始化日志系统
    init_logger()
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="深度学习模型训练与预测工具")
    
    # 基础参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train（训练）或 predict（预测）')
    
    # 其他参数会被直接传递给相应的脚本
    # 使用parse_known_args获取未识别的参数，这些将被传递给子脚本
    args, unknown_args = parser.parse_known_args()
    
    # 获取当前脚本所在目录，确保能正确找到子脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据模式调用相应的脚本
    if args.mode == 'train':
        train_script = os.path.join(script_dir, 'train.py')
        cmd = [sys.executable, train_script] + unknown_args
        info(f"🚀 启动训练模式，调用: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif args.mode == 'predict':
        predict_script = os.path.join(script_dir, 'predict.py')
        cmd = [sys.executable, predict_script] + unknown_args
        info(f"🚀 启动预测模式，调用: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    else:
        error(f"不支持的模式: {args.mode}，请选择'train'/'predict'")
        raise ValueError(f"不支持的模式: {args.mode}，请选择'train'/'predict'")


if __name__ == "__main__":
    main()
