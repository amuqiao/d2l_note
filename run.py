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
from src.utils.model_registry import ModelRegistry
from src.predictor.predictor import Predictor
from src.utils.data_utils import DataLoader
from src.trainer.trainer import Trainer
from src.utils.common_train import train_model_common

# 导入模型并注册
from src.models.lenet import LeNet, LeNetBatchNorm
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet
from src.models.dense_net import DenseNet  # 使用装饰器自动注册的模型
from src.models.mlp import MLP  # 使用装饰器自动注册的模型

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
        model_type = kwargs.get("model_type")  # 默认使用ResNet
        num_epochs = kwargs.get("num_epochs")
        lr = kwargs.get("lr")
        batch_size = kwargs.get("batch_size")
        input_size = kwargs.get("input_size")
        resize = kwargs.get("resize")

        # 构建配置参数
        config = {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "resize": resize,
            "input_size": input_size,
            "root_dir": kwargs.get("root_dir")
        }
        
        # 使用公共训练方法训练模型
        save_every_epoch = kwargs.get("save_every_epoch", False)
        enable_visualization = kwargs.get("enable_visualization", True)
        result = train_model_common(model_type, config, enable_visualization, save_every_epoch)
        
        # 获取训练结果
        if result["success"]:
            run_dir = result["run_dir"]
            best_acc = result["best_accuracy"]
        else:
            raise RuntimeError(f"模型训练失败: {result['error']}")

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
            if predictor.config["model_name"] == "AlexNet" or predictor.config["model_name"] == "VGG":
                resize = 224  # AlexNet和VGG都需要224x224输入
            elif predictor.config["model_name"] == "GoogLeNet":
                resize = 96   # GoogLeNet需要96x96输入

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

# 初始化模型注册中心
# 注意：所有模型已在对应模型文件中通过装饰器注册了配置
# 包括：LeNet、AlexNet、VGG、NIN、GoogLeNet、ResNet、DenseNet

# 为了向后兼容，保留MODEL_DEFAULT_CONFIGS变量
MODEL_DEFAULT_CONFIGS = {}
for model_name in ModelRegistry.list_models():
    if ModelRegistry.is_registered(model_name):
        try:
            MODEL_DEFAULT_CONFIGS[model_name] = ModelRegistry.get_config(model_name)
        except ValueError:
            pass

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="深度学习模型训练与预测工具")
    
    # 获取已注册的模型列表，用于动态填充参数选项
    registered_models = ModelRegistry.list_models()
    
    # 基础参数
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='运行模式: train（训练）或 predict（预测）')
    
    # 训练模式参数
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
            'batch_size': args.batch_size if args.batch_size is not None else 256,  # 默认批次大小为256
            'root_dir': args.root_dir
        })
    
    # 启动主函数
    main(**call_args)
