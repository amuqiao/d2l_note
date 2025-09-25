from typing import List
from src.model_visualization.data_models import ModelInfoData
from src.utils.log_utils import get_logger

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class TimestrideModelComparisonVisualizer:
    """Timestride模型比较可视化器（这是一个特殊的可视化器，不遵循标准接口）"""
    
    def __init__(self):
        self.model_infos = []
    
    def add_model_info(self, model_info: ModelInfoData):
        """添加模型信息到比较列表"""
        self.model_infos.append(model_info)
    
    def visualize(self, show: bool = True) -> None:
        """执行模型比较可视化"""
        if len(self.model_infos) < 2:
            print("比较功能需要至少两个模型数据")
            return
        
        # 基础信息
        print("=== Timestride模型比较 ===")
        print(f"比较模型数量: {len(self.model_infos)}")
        
        # 按测试准确率排序
        data_sorted = sorted(self.model_infos, key=lambda x: x.metrics.get("final_test_accuracy", 0) if hasattr(x, 'metrics') else 0, reverse=True)
        
        # 关键指标比较表
        print("\n--- 关键指标比较 ---")
        # 表头
        print(f"{'排名':<5}{'模型类型':<20}{'最终测试准确率':<20}{'最佳测试准确率':<20}{'最终测试损失':<20}")
        print("-" * 85)
        
        # 按顺序显示各个模型的指标
        for i, model_data in enumerate(data_sorted, 1):
            model_type = model_data.model_type[:18] + "..." if len(model_data.model_type) > 20 else model_data.model_type
            metrics = getattr(model_data, 'metrics', {})
            final_test_acc = metrics.get("final_test_accuracy", 0)
            best_test_acc = metrics.get("best_test_accuracy", 0)
            final_test_loss = metrics.get("final_test_loss", 0)
            
            print(f"{i:<5}{model_type:<20}{final_test_acc:.2%}{'':<2}{best_test_acc:.2%}{'':<2}{final_test_loss:.6f}")
        
        # 参数比较
        print("\n--- 参数比较 ---")
        # 收集所有唯一参数名
        all_params = set()
        for model_data in self.model_infos:
            if hasattr(model_data, 'params') and model_data.params:
                all_params.update(model_data.params.keys())
        
        # 选择重要的参数进行比较
        important_params = ["model_name", "task_name", "batch_size", "learning_rate", "train_epochs", "device"]
        params_to_compare = [p for p in important_params if p in all_params]
        
        # 如果重要参数不足，则补充其他参数
        if len(params_to_compare) < 3 and len(all_params) > len(params_to_compare):
            other_params = [p for p in all_params if p not in important_params]
            params_to_compare.extend(other_params[:3 - len(params_to_compare)])
        
        # 展示参数比较
        for param in params_to_compare:
            values = []
            for i, model_data in enumerate(data_sorted, 1):
                params = getattr(model_data, 'params', {})
                value = params.get(param, "N/A")
                values.append(f"模型{i}: {value}")
            print(f"{param}:")
            print("  " + " | ".join(values))
        
        return None
