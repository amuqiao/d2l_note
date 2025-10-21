from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import re
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils.log_utils import get_logger

# 导入基类以避免循环导入问题
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class TrainingMetricsVisualizer(BaseModelInfoVisualizer):
    """训练指标可视化器：展示训练性能指标"""
    
    def support(self, model_info: ModelInfoData) -> bool:
        # 仅支持有metrics信息的训练任务
        return model_info.type == "run" and "final_test_acc" in model_info.metrics
    
    def visualize(self, model_info: ModelInfoData, show: bool = True) -> plt.Figure:
        try:
            # 使用工具注册中心设置中文字体
            ToolRegistry.call("setup_font")
            
            metrics = model_info.metrics
            
            # 创建图表
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 绘制准确率指标
            accuracy_data = {
                "Final Train Acc": metrics.get("final_train_acc", 0),
                "Final Test Acc": metrics.get("final_test_acc", 0),
                "Best Test Acc": metrics.get("best_test_acc", 0)
            }
            
            axes[0].bar(accuracy_data.keys(), accuracy_data.values(), color=['blue', 'green', 'red'])
            axes[0].set_ylabel('准确率')
            axes[0].set_title('模型准确率指标')
            axes[0].set_ylim(0, 1)
            
            # 在柱状图上标注数值
            for i, v in enumerate(accuracy_data.values()):
                axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center')
            
            # 绘制训练效率指标
            efficiency_data = {
                "Training Time": metrics.get("total_training_time", "0s"),
                "Samples/Second": metrics.get("samples_per_second", "0")
            }
            
            # 提取数值用于可视化
            time_value = 0
            if isinstance(efficiency_data["Training Time"], str):
                # 尝试解析时间字符串（如 "6.26s"）
                time_match = re.search(r'([\d.]+)', efficiency_data["Training Time"])
                if time_match:
                    time_value = float(time_match.group(1))
            
            samples_value = 0
            if isinstance(efficiency_data["Samples/Second"], str):
                # 尝试解析样本数字符串
                samples_match = re.search(r'([\d.]+)', efficiency_data["Samples/Second"])
                if samples_match:
                    samples_value = float(samples_match.group(1))
            
            # 为效率指标创建双Y轴图表
            ax2 = axes[1]
            
            # 左侧Y轴：训练时间
            time_bars = ax2.bar(0, time_value, color='purple', width=0.3, label='训练时间 (秒)')
            ax2.set_ylabel('训练时间 (秒)', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            # 右侧Y轴：每秒处理样本数
            ax2_right = ax2.twinx()
            samples_bars = ax2_right.bar(0.5, samples_value, color='orange', width=0.3, label='每秒处理样本数')
            ax2_right.set_ylabel('每秒处理样本数', color='orange')
            ax2_right.tick_params(axis='y', labelcolor='orange')
            
            # 设置X轴
            ax2.set_xticks([0, 0.5])
            ax2.set_xticklabels(['训练时间', '样本处理速度'])
            ax2.set_title('训练效率指标')
            
            # 在柱状图上标注数值
            for bar in time_bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}s', ha='center', va='bottom', color='purple')
            
            for bar in samples_bars:
                height = bar.get_height()
                ax2_right.text(bar.get_x() + bar.get_width()/2., height + 100,
                            f'{height:.0f}', ha='center', va='bottom', color='orange')
            
            # 设置整体标题
            fig.suptitle(f"训练性能指标 - {model_info.model_type}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"绘制训练指标可视化失败: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer