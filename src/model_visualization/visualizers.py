from abc import ABC, abstractmethod
from typing import List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import os
from abc import ABC, abstractmethod
from prettytable import PrettyTable
from src.model_visualization.data_models import MetricData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils import get_logger


# 初始化日志器（可视化器专属日志，可选）
logger = get_logger(name=__name__, log_file="logs/visualizer.log", global_level="INFO")


# ==================================================
# 可视化层：负责将标准化数据模型转换为可视化结果
# ==================================================
class BaseVisualizer(ABC):

    @abstractmethod
    def support(self, metric_data: MetricData) -> bool:
        """判断当前可视化器是否支持该指标数据"""
        pass

    @abstractmethod
    def visualize(self, metric_data: MetricData, show: bool = True) -> Any:
        """绘制可视化结果（返回绘图对象）"""
        pass


class VisualizerRegistry:
    """可视化器注册中心：管理所有可视化器，实现自动匹配"""

    _visualizers: List[BaseVisualizer] = []

    @classmethod
    def register(cls, visualizer: BaseVisualizer):
        """注册可视化器实例"""
        cls._visualizers.append(visualizer)

    @classmethod
    def get_matched_visualizer(
        cls, metric_data: MetricData
    ) -> Optional[BaseVisualizer]:
        """根据指标数据匹配最合适的可视化器"""
        for vis in cls._visualizers:
            if vis.support(metric_data):
                return vis
        return None

    @classmethod
    def draw(cls, metric_data: MetricData, show: bool = True) -> Any:
        """自动绘制可视化结果的入口方法"""
        vis = cls.get_matched_visualizer(metric_data)
        if not vis:
            logger.warning(f"未找到匹配的可视化器: {metric_data.metric_type}")
            return None

        try:
            return vis.visualize(metric_data, show)
        except Exception as e:
            logger.error(f"可视化失败 {metric_data.source_path}: {str(e)}")
            return None


# ==================================================
# 具体可视化器实现
# ==================================================
class CurveVisualizer(BaseVisualizer):
    """曲线可视化器：展示训练过程中的loss和acc变化曲线"""

    def support(self, metric_data: MetricData) -> bool:
        return metric_data.metric_type == "epoch_curve"

    def visualize(self, metric_data: MetricData, show: bool = True) -> Any:
        try:
            from d2l import torch as d2l  # 延迟导入，避免不必要的依赖加载

            # 使用工具注册中心设置中文字体
            ToolRegistry.call("setup_font")
            data = metric_data.data

            animator = d2l.Animator(
                xlabel="迭代周期",
                xlim=[1, len(data["epochs"])],
                legend=["训练损失", "训练准确率", "测试准确率"],
                title=f"训练曲线 (来源: {os.path.basename(metric_data.source_path)})",
            )

            for i in range(len(data["epochs"])):
                animator.add(
                    data["epochs"][i],
                    (data["train_loss"][i], data["train_acc"][i], data["test_acc"][i]),
                )

            if show:
                plt.show()
            return animator
        except Exception as e:
            logger.error(f"绘制曲线可视化失败: {str(e)}")
            return None


class ConfusionMatrixVisualizer(BaseVisualizer):
    """混淆矩阵可视化器：展示模型分类结果的混淆矩阵"""

    def support(self, metric_data: MetricData) -> bool:
        return metric_data.metric_type == "confusion_matrix"

    def visualize(self, metric_data: MetricData, show: bool = True) -> plt.Figure:
        try:
            # 使用工具注册中心设置中文字体
            ToolRegistry.call("setup_font")
            data = metric_data.data
            matrix = np.array(data["matrix"])
            classes = data["classes"]

            # 确保矩阵和类别数量匹配
            if len(matrix) != len(classes) or len(matrix[0]) != len(classes):
                logger.error("混淆矩阵维度与类别数量不匹配")
                return None

            # 绘制混淆矩阵
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, cmap="Blues")

            # 设置坐标轴
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            ax.set_xlabel("预测类别")
            ax.set_ylabel("真实类别")
            ax.set_title(f"混淆矩阵 (准确率: {data['accuracy']:.4f})")

            # 添加数值标注
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

            plt.colorbar(im)

            if show:
                plt.show()
            return fig
        except Exception as e:
            logger.error(f"绘制混淆矩阵可视化失败: {str(e)}")
            return None


class LabelStatsVisualizer(BaseVisualizer):
    """标签统计可视化器：使用prettytable输出标签统计信息表格"""

    def support(self, metric_data: MetricData) -> bool:
        return metric_data.metric_type == "label_stats"

    def visualize(self, metric_data: MetricData, show: bool = True) -> PrettyTable:
        try:
            data = metric_data.data
            dataset_type = data["dataset_type"]
            total_samples = data["total_samples"]
            label_counts = data["label_counts"]
            named_label_counts = data["named_label_counts"]
            
            # 创建表格
            table = PrettyTable()
            table.title = f"{dataset_type.upper()} 数据集标签统计（共{total_samples}个样本）"
            table.field_names = ["标签ID", "标签名称", "样本数量", "占比"]
            
            # 构建标签ID到标签名称的映射（通过匹配计数值）
            # 创建一个反向映射：{count: [label_names]}
            count_to_names = {}
            for name, count in named_label_counts.items():
                if count not in count_to_names:
                    count_to_names[count] = []
                count_to_names[count].append(name)
            
            # 填充表格数据
            for label_id, count in label_counts.items():
                # 查找与当前计数匹配的标签名称
                label_name = "未知"
                if count in count_to_names and len(count_to_names[count]) > 0:
                    # 如果有多个标签名称具有相同的计数，取第一个
                    label_name = count_to_names[count][0]
                    # 从映射中移除已使用的名称，避免重复匹配
                    if len(count_to_names[count]) > 1:
                        count_to_names[count] = count_to_names[count][1:]
                    else:
                        del count_to_names[count]
                
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                table.add_row([label_id, label_name, count, f"{percentage:.2f}%"])
            
            # 美化表格
            table.align = "l"
            table.align["样本数量"] = "r"
            table.align["占比"] = "r"
            
            # 显示表格
            if show:
                print(table)
            
            return table
        except Exception as e:
            logger.error(f"绘制标签统计可视化失败: {str(e)}")
            return None


# ------------------------------
# 自动注册可视化器
# ------------------------------
VisualizerRegistry.register(CurveVisualizer())
VisualizerRegistry.register(ConfusionMatrixVisualizer())
VisualizerRegistry.register(LabelStatsVisualizer())
