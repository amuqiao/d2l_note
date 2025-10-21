import time
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.utils.log_utils.log_utils import get_logger

# 导入基类以避免循环导入问题
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class TimestrideModelSummaryVisualizer(BaseModelInfoVisualizer):
    """Timestride模型摘要可视化器"""
    
    def support(self, model_info: ModelInfoData) -> bool:
        return model_info.type in ["model", "run"]
    
    def visualize(self, model_info: ModelInfoData, show: bool = True) -> None:
        # 基础信息
        print("=== Timestride模型摘要 ===")
        print(f"路径: {model_info.path}")
        print(f"类型: {model_info.model_type}")
        print(f"创建时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_info.timestamp))}")
        
        # 参数信息
        print("\n--- 模型参数 ---")
        if model_info.params:
            # 提取重要的参数进行展示
            important_params = [
                "model_name", "task_name", "batch_size", "learning_rate", 
                "train_epochs", "device", "num_class", "total_params"
            ]
            for param in important_params:
                if param in model_info.params:
                    value = model_info.params[param]
                    print(f"{param}: {value}")
            
            # 如果有其他参数，也展示出来
            other_params = [p for p in model_info.params if p not in important_params]
            if other_params:
                print("\n其他参数:")
                for param in other_params:
                    print(f"  {param}: {model_info.params[param]}")
        else:
            print("暂无参数信息")
        
        # 性能指标
        print("\n--- 性能指标 ---")
        if model_info.metrics:
            metric_order = [
                "final_train_loss", "final_val_loss", "final_test_loss",
                "final_val_accuracy", "final_test_accuracy",
                "best_test_accuracy", "num_epochs"
            ]
            
            for metric in metric_order:
                if metric in model_info.metrics:
                    value = model_info.metrics[metric]
                    # 格式化指标名称，使其更易读
                    readable_name = metric.replace("_", " ").title()
                    print(f"{readable_name}: {value}")
        else:
            print("暂无性能指标信息")
        
        return None