import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Type


# 导入自定义日志模块
from src.utils.log_utils.log_utils import get_logger

# 导入工具注册中心
from src.helper_utils.helper_tools_registry import ToolRegistry
# 导入MetricData类
from src.model_helper_utils.metric_parsers import MetricData

from src.model_helper_utils.visualizers import BaseVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_analysis.log", global_level="INFO")


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
                title=f"训练曲线 (来源: {os.path.basename(metric_data.source_path)})"
            )
            
            for i in range(len(data["epochs"])):
                animator.add(
                    data["epochs"][i],
                    (data["train_loss"][i], data["train_acc"][i], data["test_acc"][i])
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