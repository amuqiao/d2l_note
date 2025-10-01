"""
模型指标可视化器：使用prettytable展示模型训练指标

功能：将模型训练的各项指标（标量、曲线等）可视化为美观易读的表格形式
"""
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from prettytable import PrettyTable
from src.model_show_v2.data_models import ModelInfoData, MetricData
from src.utils.log_utils.log_utils import get_logger
from .base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry

logger = get_logger(name=__name__)


@ModelVisualizerRegistry.register(namespace="metrics")
class MetricsVisualizer(BaseModelVisualizer):
    """模型指标可视化器：使用prettytable展示模型训练指标"""
    
    # 设置较高优先级
    priority: int = 70
    
    def __init__(self):
        """初始化模型指标可视化器"""
        pass
    
    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        """判断是否支持该模型信息的可视化
        
        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"
        
        Returns:
            bool: 是否支持该模型信息
        """
        # 检查是否有指标数据
        if not model_info.metric_list:
            return False
        
        # 检查是否有metrics.json文件关联
        if model_info.path and os.path.exists(os.path.join(model_info.path, "metrics.json")):
            return True
        
        # 检查指标数据类型
        for metric in model_info.metric_list:
            if "metrics" in metric.source_path.lower():
                return True
        
        return False
    
    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        """将模型指标可视化为表格形式
        
        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"
        
        Returns:
            Dict[str, Any]: 可视化结果，包含表格对象和显示文本
        """
        try:
            # 创建主结果字典
            result = {
                "success": True,
                "message": "模型指标可视化成功",
                "tables": {}
            }
            
            # 添加优化的指标表格（用户要求的格式）
            optimized_table = self._create_optimized_metrics_table([model_info])
            result["tables"]["optimized_metrics"] = optimized_table
            result["text"] = str(optimized_table)
            
            return result
            
        except Exception as e:
            logger.error(f"模型指标可视化失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型指标可视化失败: {str(e)}"
            }
    
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        """比较多个模型的指标信息
        
        Args:
            model_infos: 模型信息数据列表
            namespace: 命名空间，默认为"default"
        
        Returns:
            Dict[str, Any]: 比较可视化结果
        """
        try:
            if len(model_infos) < 2:
                return {
                    "success": False,
                    "message": "比较需要至少2个模型指标信息"
                }
                
            # 创建比较结果字典
            result = {
                "success": True,
                "message": "模型指标比较成功",
                "tables": {}
            }
            
            # 创建优化的多模型比较表格
            optimized_compare_table = self._create_optimized_metrics_table(model_infos)
            result["tables"]["optimized_compare"] = optimized_compare_table
            result["text"] = str(optimized_compare_table)
            
            return result
            
        except Exception as e:
            logger.error(f"模型指标比较失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型指标比较失败: {str(e)}"
            }
    
    def _create_optimized_metrics_table(self, model_infos: List[ModelInfoData]) -> PrettyTable:
        """创建优化的模型指标表格，按照用户要求的格式显示特定字段"""
        table = PrettyTable()
        
        # 设置表格标题
        if len(model_infos) == 1:
            table.title = f"模型指标详情 ({model_infos[0].name})"
        else:
            table.title = "模型指标比较"
        
        # 设置表头
        headers = ["模型名称", "final_train_loss", "final_train_acc", "final_test_acc", 
                  "best_test_acc", "total_training_time", "samples_per_second", 
                  "epoch轮次", "training_start_time", "training_end_time"]
        
        table.field_names = headers
        
        # 为每个模型添加一行数据
        for model_info in model_infos:
            row = [model_info.name]  # 模型名称
            
            # final_train_loss
            train_loss_metric = model_info.get_metric_by_name("final_train_loss")
            if train_loss_metric and "value" in train_loss_metric.data:
                row.append(self._format_float_value(train_loss_metric.data["value"]))
            else:
                row.append("-")
            
            # final_train_acc
            train_acc_metric = model_info.get_metric_by_name("final_train_acc")
            if train_acc_metric and "value" in train_acc_metric.data:
                row.append(self._format_float_value(train_acc_metric.data["value"]))
            else:
                row.append("-")
            
            # final_test_acc
            test_acc_metric = model_info.get_metric_by_name("final_test_acc")
            if test_acc_metric and "value" in test_acc_metric.data:
                row.append(self._format_float_value(test_acc_metric.data["value"]))
            else:
                row.append("-")
            
            # best_test_acc
            best_test_acc_metric = model_info.get_metric_by_name("best_test_acc")
            if best_test_acc_metric and "value" in best_test_acc_metric.data:
                row.append(self._format_float_value(best_test_acc_metric.data["value"]))
            else:
                row.append("-")
            
            # total_training_time
            total_training_time = model_info.params.get("total_training_time", "-")
            row.append(total_training_time)
            
            # samples_per_second
            samples_per_second = model_info.params.get("samples_per_second", "-")
            row.append(samples_per_second)
            
            # epoch轮次
            epoch_count = self._get_epoch_count(model_info)
            row.append(epoch_count)
            
            # training_start_time
            training_start_time = model_info.params.get("training_start_time", "-")
            row.append(training_start_time)
            
            # training_end_time
            training_end_time = model_info.params.get("training_end_time", "-")
            row.append(training_end_time)
            
            table.add_row(row)
        
        # 美化表格
        table.align["模型名称"] = "l"
        for field in headers[1:]:
            table.align[field] = "r"
        
        return table
    
    def _get_epoch_count(self, model_info: ModelInfoData) -> int:
        """获取模型训练的epoch轮次"""
        # 尝试从epoch_metrics相关的指标中获取轮次信息
        for metric in model_info.metric_list:
            # 检查是否有epochs字段
            if "epochs" in metric.data:
                return len(metric.data["epochs"])
            # 检查是否有values字段（曲线数据长度可能代表epoch数量）
            elif "values" in metric.data:
                return len(metric.data["values"])
        
        # 如果没有找到，尝试从params中获取
        if "epochs" in model_info.params:
            return model_info.params["epochs"]
        
        # 默认返回-1表示未找到
        return -1
    
    def _format_timestamp(self, timestamp: float) -> str:
        """格式化时间戳为可读日期时间"""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)
    
    def _format_float_value(self, value: float) -> str:
        """格式化浮点数值为合适的字符串表示"""
        # 根据数值大小选择合适的精度
        if abs(value) >= 100:
            return f"{value:.1f}"
        elif abs(value) >= 10:
            return f"{value:.2f}"
        elif abs(value) >= 1:
            return f"{value:.3f}"
        else:
            return f"{value:.4f}"