import os
import sys
import glob
import json
import torch
from typing import Optional, Dict, Any, List
from src.utils.data_utils import DataLoader
from src.utils.file_utils import FileUtils
from src.predictor.predictor import Predictor
from src.utils.visualization import VisualizationTool

class PredictorTool:
    """预测工具类：封装模型预测相关的所有功能"""
    
    def __init__(self):
        """初始化预测工具"""
        # 设置中文字体
        VisualizationTool.setup_font()
        
    def find_latest_run_dir(self, root_dir: Optional[str] = None) -> str:
        """
        自动查找最新的训练目录
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            最新训练目录的路径
            
        Raises:
            FileNotFoundError: 当未找到训练目录时
        """
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
            return run_dir
        else:
            # 如果没找到，提供更详细的错误信息
            searched_paths = ", ".join([os.path.join(d, "run_*") for d in search_dirs])
            raise FileNotFoundError(f"未找到任何训练目录（需以'run_'开头）\n已搜索路径: {searched_paths}")
            
    def get_predictor(self, run_dir: Optional[str] = None, 
                     model_file: Optional[str] = None, 
                     model_path: Optional[str] = None) -> Predictor:
        """
        创建并返回预测器实例
        
        Args:
            run_dir: 训练目录路径
            model_file: 模型文件名
            model_path: 完整模型文件路径
            
        Returns:
            Predictor实例
        """
        if model_path:
            # 直接从模型文件路径加载
            print(f"🔍 模式：从模型文件直接加载")
            return Predictor.from_model_path(model_path)
        else:
            # 从训练目录加载（可选择模型文件）
            print(f"🔍 模式：从训练目录加载{', 自动选择最佳模型' if not model_file else f', 指定模型文件: {model_file}'}")
            return Predictor.from_run_dir(run_dir, model_file=model_file)
    
    def get_resize_for_model(self, predictor: Predictor) -> Optional[int]:
        """
        根据模型类型确定合适的resize参数
        
        Args:
            predictor: Predictor实例
            
        Returns:
            适合该模型的resize值或None
        """
        resize = None  # 默认不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet" or predictor.config["model_name"] == "VGG":
                resize = 224  # AlexNet和VGG都需要224x224输入
            elif predictor.config["model_name"] == "GoogLeNet":
                resize = 96   # GoogLeNet需要96x96输入
        return resize
        
    def list_models_in_directory(self, run_dir: str) -> None:
        """
        列出目录中的所有模型信息
        
        Args:
            run_dir: 训练目录路径
        """
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
    
    def run_prediction(self, run_dir: Optional[str] = None, 
                      model_file: Optional[str] = None, 
                      model_path: Optional[str] = None, 
                      batch_size: int = 256, 
                      num_samples: int = 10, 
                      root_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行模型预测并返回结果
        
        Args:
            run_dir: 训练目录路径
            model_file: 模型文件名
            model_path: 完整模型文件路径
            batch_size: 批次大小
            num_samples: 随机测试样本数
            root_dir: 根目录路径
            
        Returns:
            预测结果字典
        """
        # 如果未指定训练目录或模型路径，自动选择最新目录
        if not run_dir and not model_path:
            run_dir = self.find_latest_run_dir(root_dir)
        
        # 创建预测器
        predictor = self.get_predictor(run_dir, model_file, model_path)
        
        # 如果指定了目录但未指定模型文件，显示该目录下的所有模型信息
        if not model_file and not model_path:
            self.list_models_in_directory(run_dir)
        
        # 根据模型类型确定resize参数
        resize = self.get_resize_for_model(predictor)
        
        # 加载数据（使用正确的resize参数）
        _, test_iter = DataLoader.load_data(
            batch_size=batch_size, resize=resize
        )
        
        # 执行预测可视化
        predictor.visualize_prediction(test_iter, n=num_samples)
        predictor.test_random_input(num_samples=num_samples)
        
        # 返回预测结果信息
        result = {
            "success": True,
            "model_name": predictor.config["model_name"] if predictor.config else "未知",
            "run_dir": run_dir,
            "model_file": model_file,
            "model_path": model_path
        }
        
        return result