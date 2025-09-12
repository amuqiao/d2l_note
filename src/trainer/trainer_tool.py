import os
import sys
import json
import datetime
from typing import Optional, Dict, Any
from src.utils.model_registry import ModelRegistry
from src.utils.common_train import train_model_common
from src.utils.visualization import VisualizationTool
from src.predictor.predictor import Predictor
from src.utils.data_utils import DataLoader

class TrainerTool:
    """训练工具类：封装模型训练相关的所有功能"""
    
    def __init__(self):
        """初始化训练工具"""
        # 设置中文字体
        VisualizationTool.setup_font()
    
    def get_model_config(self, model_type: str, **kwargs) -> Dict[str, Any]:
        """
        获取模型的配置参数
        
        Args:
            model_type: 模型类型名称
            **kwargs: 可选的自定义配置参数
        
        Returns:
            模型配置字典
        """
        try:
            # 直接从模型注册中心获取配置
            default_config = ModelRegistry.get_config(model_type)
            
            # 创建配置字典，确保不会有None值传递给关键参数
            config = {
                "num_epochs": kwargs.get("num_epochs") if kwargs.get("num_epochs") is not None else default_config["num_epochs"],
                "lr": kwargs.get("lr") if kwargs.get("lr") is not None else default_config["lr"],
                "batch_size": kwargs.get("batch_size") if kwargs.get("batch_size") is not None else default_config["batch_size"],
                "resize": kwargs.get("resize") if kwargs.get("resize") is not None else default_config["resize"],
                "input_size": kwargs.get("input_size") if kwargs.get("input_size") is not None else default_config["input_size"],
                "root_dir": kwargs.get("root_dir")
            }
            
            # 确保num_epochs不是None
            if config["num_epochs"] is None:
                config["num_epochs"] = 10  # 默认值
                print(f"⚠️ 警告: {model_type}的num_epochs为None，使用默认值10")
            
            return config
        except ValueError:
            # 处理配置不存在的情况
            print(f"⚠️ 模型 '{model_type}' 没有默认配置，使用基础配置")
            return {
                "num_epochs": kwargs.get("num_epochs", 10),
                "lr": kwargs.get("lr", 0.01),
                "batch_size": kwargs.get("batch_size", 128),
                "input_size": kwargs.get("input_size", (1, 1, 28, 28)),
                "resize": kwargs.get("resize"),
                "root_dir": kwargs.get("root_dir")
            }
    
    def run_training(self, model_type: str, config: Dict[str, Any], 
                    enable_visualization: bool = True, 
                    save_every_epoch: bool = False) -> Dict[str, Any]:
        """
        执行模型训练并返回结果
        
        Args:
            model_type: 模型类型名称
            config: 训练配置参数
            enable_visualization: 是否启用可视化
            save_every_epoch: 是否每轮都保存模型文件
        
        Returns:
            训练结果字典
        """
        # 使用公共训练方法训练模型
        result = train_model_common(model_type, config, enable_visualization, save_every_epoch)
        
        # 处理训练结果
        if not result["success"]:
            raise RuntimeError(f"模型训练失败: {result['error']}")
        
        return result
    
    def run_post_training_prediction(self, run_dir: str, n: int = 8, num_samples: int = 10) -> None:
        """
        训练后自动进行预测可视化
        
        Args:
            run_dir: 训练目录路径
            n: 可视化样本数
            num_samples: 随机测试样本数
        """
        print(f"\n🎉 训练完成，开始预测可视化（目录: {run_dir}）")
        predictor = Predictor.from_run_dir(run_dir)
        
        # 根据模型类型确定resize参数
        resize = None  # 默认LeNet不需要resize
        if predictor.config and "model_name" in predictor.config:
            if predictor.config["model_name"] == "AlexNet":
                resize = 224  # AlexNet需要224x224输入
            elif predictor.config["model_name"] == "VGG":
                resize = 224  # VGG需要224x224输入
            elif predictor.config["model_name"] == "GoogLeNet":
                resize = 96   # GoogLeNet需要96x96输入
        
        # 重新加载测试数据（使用正确的resize参数）
        _, test_iter_pred = DataLoader.load_data(batch_size=256, resize=resize)
        
        # 执行预测可视化
        predictor.visualize_prediction(test_iter_pred, n=n)
        predictor.test_random_input(num_samples=num_samples)