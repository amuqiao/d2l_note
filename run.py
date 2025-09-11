import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import sys
import matplotlib.pyplot as plt
import os
import glob
import json
import datetime
import re
import argparse
from src.utils.visualization import VisualizationTool
from src.utils.file_utils import FileUtils
from src.utils.network_utils import NetworkUtils
from src.predictor.predictor import Predictor
from src.data.data_loader import DataLoader
from src.trainer.trainer import Trainer
from src.models.lenet import LeNet
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet

# 解决OpenMP运行时库冲突问题
# 设置环境变量允许多个OpenMP运行时库共存
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========================= 主函数入口 =========================
def main(mode="train", run_dir=None, model_file=None, **kwargs):
    """
    主函数：支持训练、预测、结果汇总三种模式
    Args:
        mode: 运行模式（train/predict/summarize）
        run_dir: 训练目录（predict模式需指定，train模式自动生成）
        model_file: 可选，指定的模型文件名
        kwargs: 模式参数（如train的num_epochs、lr等）
    """
    # 初始化字体
    VisualizationTool.setup_font()

    if mode == "train":
        # 训练模式：直接使用传入的参数（call_args已处理默认值）
        model_type = kwargs.get("model_type")  # 默认使用VGG
        num_epochs = kwargs.get("num_epochs")
        lr = kwargs.get("lr")
        batch_size = kwargs.get("batch_size")
        input_size = kwargs.get("input_size")
        resize = kwargs.get("resize")

        # 1. 创建模型
        if model_type == "LeNet":
            net = LeNet()
        elif model_type == "AlexNet":
            net = AlexNet()
        elif model_type == "VGG":
            net = VGG()
        elif model_type == "NIN":
            net = NIN()
        elif model_type == "GoogLeNet":
            net = GoogLeNet()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 测试网络结构
        NetworkUtils.test_network_shape(net, input_size=input_size)

        # 2. 加载数据
        print(f"📥 加载Fashion-MNIST（batch_size={batch_size}, resize={resize}）")
        train_iter, test_iter = d2l.load_data_fashion_mnist(
            batch_size=batch_size, resize=resize
        )

        # 3. 训练模型
        save_every_epoch = kwargs.get("save_every_epoch", False)
        trainer = Trainer(net, save_every_epoch=save_every_epoch)
        # 训练模型（启用实时可视化）
        print("🎨 启用实时可视化训练过程...")
        enable_visualization = kwargs.get("enable_visualization", True)
        run_dir, best_acc = trainer.train(
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            enable_visualization=enable_visualization,  # 启用实时可视化
        )

        # 4. 训练后自动预测可视化
        print(f"\n🎉 训练完成，开始预测可视化（目录: {run_dir}）")
        predictor = Predictor.from_run_dir(run_dir)

        # 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入

        # 重新加载测试数据（使用正确的resize参数）
        _, test_iter_pred = DataLoader.load_data(batch_size=256, resize=resize)

        # 执行预测可视化
        predictor.visualize_prediction(test_iter_pred, n=8)
        predictor.test_random_input(num_samples=10)

    elif mode == "predict":
        # 预测模式：支持多种加载方式
        # 方式1：从模型文件直接加载
        # 方式2：从训练目录加载（自动加载最佳模型）

        # 方式3：两者都不指定时，自动选择最新训练目录
        
        if not run_dir and not kwargs.get("model_path"):
            # 自动选择最新训练目录
            print("⚠️ 未指定训练目录或模型路径，自动查找最新目录...")
            
            # 获取root_dir参数，如果为None则使用当前工作目录
            root_dir = kwargs.get("root_dir")
            if root_dir is None:
                root_dir = os.getcwd()
            
            # 定义要搜索的目录列表（优先级顺序）
            search_dirs = [
                root_dir,  # 首先在root_dir目录下查找
                os.path.join(root_dir, "data")  # 然后在root_dir/data目录下查找
            ]
            
            # 存储所有找到的run_开头的目录
            all_run_dirs = []
            
            # 遍历搜索目录
            for search_dir in search_dirs:
                try:
                    if os.path.exists(search_dir) and os.path.isdir(search_dir):
                        run_dirs_in_search = [
                            os.path.join(search_dir, d)
                            for d in os.listdir(search_dir)
                            if os.path.isdir(os.path.join(search_dir, d)) and d.startswith("run_")
                        ]
                        all_run_dirs.extend(run_dirs_in_search)
                except Exception as e:
                    print(f"⚠️ 搜索目录 {search_dir} 时出错: {str(e)}")
            
            # 按修改时间排序，选择最新的目录
            if all_run_dirs:
                all_run_dirs.sort(key=os.path.getmtime, reverse=True)
                run_dir = all_run_dirs[0]
                print(f"✅ 自动选择最新训练目录: {run_dir}")
            else:
                # 如果没找到，提供更详细的错误信息
                searched_paths = ", ".join([os.path.join(d, "run_*") for d in search_dirs])
                raise FileNotFoundError(f"未找到任何训练目录（需以'run_'开头）\n已搜索路径: {searched_paths}")

        # 1. 创建预测器并确定resize参数
        if kwargs.get("model_path"):
            # 直接从模型文件路径加载
            print(f"🔍 模式：从模型文件直接加载")
            predictor = Predictor.from_model_path(kwargs["model_path"])
        else:
            # 从训练目录加载（可选择模型文件）
            print(f"🔍 模式：从训练目录加载{', 自动选择最佳模型' if not model_file else f', 指定模型文件: {model_file}'}")
            predictor = Predictor.from_run_dir(run_dir, model_file=model_file)

            # 如果指定了目录但未指定模型文件，显示该目录下的所有模型信息
            if not model_file:
                try:
                    models_info = FileUtils.list_models_in_dir(run_dir)
                    print(f"\n📋 {run_dir} 目录中的模型列表（按准确率排序）:")
                    print(f"{'序号':<4} {'文件名':<60} {'准确率':<10} {'轮次':<6}")
                    print("-" * 80)
                    for i, model_info in enumerate(models_info, 1):
                        # 标记当前加载的最佳模型
                        mark = "⭐" if i == 1 else " "
                        print(
                            f"{i:<4} {model_info['filename']:<60} {model_info['accuracy']:.4f}    {model_info['epoch']:<6} {mark}")

                    # 提示用户可以通过model_file参数指定具体模型
                    if len(models_info) > 1:
                        print(f"\n💡 提示：使用 model_file 参数可以加载特定模型，例如:")
                        print(
                            f"   main(mode='predict', run_dir='{run_dir}', model_file='{models_info[1]['filename']}')")
                except Exception as e:
                    print(f"⚠️ 列出模型文件时出错: {str(e)}")

        # 2. 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入

        # 3. 加载数据（使用正确的resize参数）
        _, test_iter = DataLoader.load_data(
            batch_size=kwargs.get("batch_size", 256), resize=resize
        )

        # 4. 执行预测可视化
        # predictor.visualize_prediction(test_iter, n=kwargs.get("n", 8))
        predictor.test_random_input(num_samples=10)

    else:
        raise ValueError(f"不支持的模式: {mode}，请选择'train'/'predict'")


# ========================= 运行入口 =========================
import argparse

# ========================= 模型默认配置 =========================
# 键：模型名称（与代码中model_type对应）
# 值：该模型的默认参数（输入尺寸、Resize、学习率、批次大小、训练轮次等）
MODEL_DEFAULT_CONFIGS = {
    "LeNet": {
        "input_size": (1, 1, 28, 28),  # (batch, channels, height, width)
        "resize": None,                # Fashion-MNIST原始尺寸28x28，无需Resize
        "lr": 0.8,                     # LeNet适合稍高学习率
        "batch_size": 256,             # 较小输入尺寸可支持更大批次
        "num_epochs": 3,              # 收敛较快，15轮足够
    },
    "AlexNet": {
        "input_size": (1, 1, 224, 224),# AlexNet需要224x224输入
        "resize": 224,                 # 加载数据时Resize到224x224
        "lr": 0.01,                    # 较大模型需较低学习率避免震荡
        "batch_size": 128,             # 224x224输入占用显存较高，批次减小
        "num_epochs": 30,              # 训练较慢，10轮平衡效果与时间
    },
    "VGG": {
        "input_size": (1, 1, 224, 224),# VGG同样需要224x224输入
        "resize": 224,
        "lr": 0.05,                   # 更深模型需更低学习率
        "batch_size": 128,              # VGG参数量大，显存占用更高
        "num_epochs": 10,               # 训练耗时久，8轮兼顾效果
    },
    "NIN": {
        "input_size": (1, 1, 224, 224), # NIN需要224x224输入
        "resize": 224,                  # 加载数据时Resize到224x224
        "lr": 0.1,                      # 参考note.py中的设置
        "batch_size": 128,              # 参考note.py中的设置
        "num_epochs": 10,               # 参考note.py中的设置
    },
    "GoogLeNet": {
        "input_size": (1, 1, 96, 96),   # GoogLeNet需要96x96输入
        "resize": 96,                   # 加载数据时Resize到96x96
        "lr": 0.1,                      # 参考note.py中的设置
        "batch_size": 128,              # 参考note.py中的设置
        "num_epochs": 20,               # 参考note.py中的设置
    }
}

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="深度学习模型训练与预测工具")
    
    # 基础参数
    
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict'],
                        help='运行模式: train（训练）或 predict（预测）')
    
    # 训练模式参数
    parser.add_argument('--model_type', type=str, default='LeNet', choices=['LeNet', 'AlexNet', 'VGG', 'NIN', 'GoogLeNet'],
                        help='模型类型: LeNet、AlexNet或VGG')

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
    
    # 预测模式参数
    parser.add_argument('--run_dir', type=str, default=None,
                        help='训练目录路径（推荐方式）')
    parser.add_argument('--model_file', type=str, default=None,
                        help='可选，指定要加载的模型文件名，不指定则自动加载该目录下的最佳模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='完整模型文件路径（最高优先级，设置后会忽略run_dir和model_file）')
    parser.add_argument('--n', type=int, default=8,
                        help='可视化样本数')
    
    # 解析参数
    args = parser.parse_args()
    
    # 准备调用参数
    call_args = {'mode': args.mode}
    
    # 根据模式添加特定参数
    if args.mode == 'train':
        # 获取当前模型类型的默认配置
        default_config = MODEL_DEFAULT_CONFIGS[args.model_type]
        
        # 当参数为None时，使用默认配置的值
        call_args.update({
            'num_epochs': args.num_epochs if args.num_epochs is not None else default_config["num_epochs"],
            'lr': args.lr if args.lr is not None else default_config["lr"],
            'batch_size': args.batch_size if args.batch_size is not None else default_config["batch_size"],
            'input_size': args.input_size if args.input_size is not None else default_config["input_size"],
            'resize': args.resize if args.resize is not None else default_config["resize"],
            'model_type': args.model_type,
            'save_every_epoch': args.save_every_epoch,
            'enable_visualization': not args.disable_visualization,
            'n': args.n,
            'root_dir': args.root_dir
        })
    else:  # predict模式
        call_args.update({
            'run_dir': args.run_dir,
            'model_file': args.model_file,
            'model_path': args.model_path,
            'n': args.n,
            'batch_size': args.batch_size,
            'root_dir': args.root_dir
        })
    
    # 启动主函数
    main(**call_args)
