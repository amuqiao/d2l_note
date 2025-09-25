import os
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils import get_logger

# 导入基类以避免循环导入问题
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class TimestrideTrainingMetricsVisualizer(BaseModelInfoVisualizer):
    """Timestride训练指标可视化器"""
    
    def support(self, model_info: ModelInfoData) -> bool:
        return model_info.type == "run" and hasattr(model_info, 'metrics') and model_info.metrics
    
    def visualize(self, model_info: ModelInfoData, show: bool = True) -> None:
        # 基础信息
        print("=== Timestride训练指标 ===")
        print(f"路径: {model_info.path}")
        print(f"模型: {model_info.model_type}")
        print(f"训练轮数: {model_info.metrics.get('num_epochs', 0)}")
        
        # 关键指标摘要
        print("\n--- 关键指标摘要 ---")
        key_metrics = [
            ("最终训练损失", "final_train_loss"),
            ("最终验证损失", "final_val_loss"),
            ("最终测试损失", "final_test_loss"),
            ("最终验证准确率", "final_val_accuracy"),
            ("最终测试准确率", "final_test_accuracy"),
            ("最佳测试准确率", "best_test_accuracy")
        ]
        
        for display_name, metric_key in key_metrics:
            if metric_key in model_info.metrics:
                value = model_info.metrics[metric_key]
                # 对于准确率，乘以100并保留2位小数
                if "accuracy" in metric_key:
                    print(f"{display_name}: {value:.2%}")
                else:
                    print(f"{display_name}: {value:.6f}")
        
        # 尝试从原始指标文件读取完整的训练曲线数据
        raw_metrics_path = os.path.join(model_info.path, "training_metrics.json")
        if os.path.exists(raw_metrics_path):
            try:
                from src.model_visualization.data_access import DataAccessor
                raw_metrics = DataAccessor.read_file(raw_metrics_path)
                if raw_metrics and isinstance(raw_metrics, list):
                    # 检查是否有足够的训练轮次数据进行曲线展示
                    if len(raw_metrics) > 1:
                        # 计算损失和准确率的趋势
                        losses = [(epoch.get("train_loss", 0), epoch.get("vali_loss", 0), epoch.get("test_loss", 0)) for epoch in raw_metrics]
                        accuracies = [(epoch.get("val_accuracy", 0), epoch.get("test_accuracy", 0)) for epoch in raw_metrics]
                        
                        # 简单的文本图形展示趋势
                        print("\n--- 训练趋势摘要 ---")
                        
                        # 损失趋势
                        train_losses = [l[0] for l in losses if l[0] != 0]
                        if train_losses:
                            loss_change = train_losses[-1] - train_losses[0]
                            trend = "下降" if loss_change < 0 else "上升" if loss_change > 0 else "保持不变"
                            print(f"训练损失: {trend} ({abs(loss_change):.6f})")
                        
                        # 准确率趋势
                        test_accuracies = [a[1] for a in accuracies if a[1] != 0]
                        if test_accuracies:
                            acc_change = test_accuracies[-1] - test_accuracies[0]
                            trend = "上升" if acc_change > 0 else "下降" if acc_change < 0 else "保持不变"
                            print(f"测试准确率: {trend} ({abs(acc_change):.2%})")
            except Exception as e:
                logger.error(f"读取Timestride原始训练指标文件失败: {str(e)}")
        
        return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer