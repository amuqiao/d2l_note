from typing import Optional
import os
from abc import ABC, abstractmethod
from prettytable import PrettyTable
from src.model_visualization.data_models import ModelInfoData
from src.utils.log_utils.log_utils import get_logger

# 导入基类以避免循环导入问题
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class ModelSummaryVisualizer(BaseModelInfoVisualizer):
    """模型摘要可视化器：展示模型基本信息和参数"""
    
    def support(self, model_info: ModelInfoData) -> bool:
        return True  # 支持所有类型的ModelInfoData
    
    def visualize(self, model_info: ModelInfoData, show: bool = True) -> PrettyTable:
        try:
            # 创建表格
            table = PrettyTable()
            
            # 根据类型设置标题
            if model_info.type == "run":
                title = f"训练任务摘要 ({os.path.basename(model_info.path)})"
            else:
                title = f"模型文件摘要 ({os.path.basename(model_info.path)})"
            
            table.title = title
            table.field_names = ["属性", "值"]
            
            # 添加基本信息
            table.add_row(["类型", "训练任务" if model_info.type == "run" else "模型文件"])
            table.add_row(["路径", model_info.path])
            table.add_row(["模型类型", model_info.model_type])
            table.add_row(["时间戳", f"{model_info.timestamp}"])
            table.add_row(["命名空间", model_info.namespace])
            
            # 添加模型参数
            for param_name, param_value in model_info.params.items():
                table.add_row([f"参数: {param_name}", param_value])
            
            # 美化表格
            table.align = "l"
            
            # 显示表格
            if show:
                print(table)
            
            return table
        except Exception as e:
            logger.error(f"绘制模型摘要可视化失败: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer