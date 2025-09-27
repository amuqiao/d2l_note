"""
模型指标可视化器：使用prettytable展示模型训练指标

功能：将模型训练的各项指标（标量、曲线等）可视化为美观易读的表格形式
"""
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from prettytable import PrettyTable
from src.model_show_v2.data_models import ModelInfoData, MetricData
from src.utils.log_utils import get_logger
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
            
            # 添加基本模型信息表格
            basic_table = self._create_basic_info_table(model_info)
            result["tables"]["basic_info"] = basic_table
            result["text"] = str(basic_table)
            
            # 添加标量指标表格
            if model_info.get_metrics_by_type("scalar"):
                scalar_table = self._create_scalar_metrics_table(model_info)
                result["tables"]["scalar_metrics"] = scalar_table
                result["text"] += "\n\n" + str(scalar_table)
            
            # 添加曲线指标摘要表格
            if model_info.get_metrics_by_type("curve"):
                curve_table = self._create_curve_metrics_summary_table(model_info)
                result["tables"]["curve_metrics_summary"] = curve_table
                result["text"] += "\n\n" + str(curve_table)
            
            # 添加训练信息表格
            if model_info.params:
                training_table = self._create_training_info_table(model_info)
                if training_table:
                    result["tables"]["training_info"] = training_table
                    result["text"] += "\n\n" + str(training_table)
            
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
            
            # 创建模型基本信息比较表格
            basic_compare_table = self._create_basic_info_compare_table(model_infos)
            result["tables"]["basic_compare"] = basic_compare_table
            result["text"] = str(basic_compare_table)
            
            # 创建标量指标比较表格
            scalar_compare_table = self._create_scalar_metrics_compare_table(model_infos)
            if scalar_compare_table:
                result["tables"]["scalar_compare"] = scalar_compare_table
                result["text"] += "\n\n" + str(scalar_compare_table)
            
            # 创建曲线指标关键值比较表格
            curve_key_compare_table = self._create_curve_key_values_compare_table(model_infos)
            if curve_key_compare_table:
                result["tables"]["curve_key_compare"] = curve_key_compare_table
                result["text"] += "\n\n" + str(curve_key_compare_table)
            
            return result
            
        except Exception as e:
            logger.error(f"模型指标比较失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型指标比较失败: {str(e)}"
            }
    
    def _create_basic_info_table(self, model_info: ModelInfoData) -> PrettyTable:
        """创建模型基本信息表格"""
        table = PrettyTable()
        table.title = f"模型指标概览 ({model_info.name})"
        table.field_names = ["属性", "值"]
        
        # 添加基本信息
        table.add_row(["模型名称", model_info.name])
        table.add_row(["模型类型", model_info.model_type])
        table.add_row(["框架", model_info.framework])
        table.add_row(["任务类型", model_info.task_type])
        table.add_row(["时间戳", self._format_timestamp(model_info.timestamp)])
        table.add_row(["指标数量", len(model_info.metric_list)])
        table.add_row(["标量指标", len(model_info.get_metrics_by_type("scalar"))])
        table.add_row(["曲线指标", len(model_info.get_metrics_by_type("curve"))])
        
        # 美化表格
        table.align["属性"] = "l"
        table.align["值"] = "l"
        
        return table
    
    def _create_scalar_metrics_table(self, model_info: ModelInfoData) -> PrettyTable:
        """创建标量指标表格"""
        table = PrettyTable()
        table.title = "标量指标详情"
        table.field_names = ["指标名称", "值", "单位", "描述"]
        
        # 获取并排序标量指标
        scalar_metrics = sorted(model_info.get_metrics_by_type("scalar"), key=lambda m: m.name)
        
        # 添加标量指标数据
        for metric in scalar_metrics:
            value = metric.data.get("value", "-")
            unit = metric.data.get("unit", "")
            # 格式化数值，保留合适的小数位数
            if isinstance(value, float):
                value = self._format_float_value(value)
            
            table.add_row([metric.name, value, unit, metric.description])
        
        # 美化表格
        table.align["指标名称"] = "l"
        table.align["值"] = "r"
        table.align["单位"] = "c"
        table.align["描述"] = "l"
        
        return table
    
    def _create_curve_metrics_summary_table(self, model_info: ModelInfoData) -> PrettyTable:
        """创建曲线指标摘要表格"""
        table = PrettyTable()
        table.title = "曲线指标摘要"
        table.field_names = ["指标名称", "数据点数量", "最大值", "最小值", "平均值", "单位"]
        
        # 获取并排序曲线指标
        curve_metrics = sorted(model_info.get_metrics_by_type("curve"), key=lambda m: m.name)
        
        # 添加曲线指标摘要数据
        for metric in curve_metrics:
            values = metric.data.get("values", [])
            # 过滤掉None值
            valid_values = [v for v in values if v is not None]
            count = len(valid_values)
            
            if count > 0:
                max_val = max(valid_values)
                min_val = min(valid_values)
                avg_val = sum(valid_values) / count
                
                # 格式化数值
                max_val = self._format_float_value(max_val)
                min_val = self._format_float_value(min_val)
                avg_val = self._format_float_value(avg_val)
            else:
                max_val = min_val = avg_val = "-"
            
            unit = metric.data.get("unit", "")
            
            table.add_row([metric.name, count, max_val, min_val, avg_val, unit])
        
        # 美化表格
        for field in table.field_names:
            if field in ["最大值", "最小值", "平均值"]:
                table.align[field] = "r"
            else:
                table.align[field] = "l"
        
        return table
    
    def _create_training_info_table(self, model_info: ModelInfoData) -> Optional[PrettyTable]:
        """创建训练信息表格"""
        # 检查是否有训练相关信息
        training_keys = ["total_training_time", "samples_per_second", 
                        "training_start_time", "training_end_time"]
        
        has_training_info = any(key in model_info.params for key in training_keys)
        if not has_training_info:
            return None
        
        table = PrettyTable()
        table.title = "训练信息"
        table.field_names = ["属性", "值"]
        
        # 添加训练时间
        if "total_training_time" in model_info.params:
            table.add_row(["总训练时间", model_info.params["total_training_time"]])
        
        # 添加每秒处理样本数
        if "samples_per_second" in model_info.params:
            table.add_row(["每秒处理样本数", model_info.params["samples_per_second"]])
        
        # 添加训练开始和结束时间
        if "training_start_time" in model_info.params:
            table.add_row(["训练开始时间", model_info.params["training_start_time"]])
        if "training_end_time" in model_info.params:
            table.add_row(["训练结束时间", model_info.params["training_end_time"]])
        
        # 美化表格
        table.align["属性"] = "l"
        table.align["值"] = "l"
        
        return table
    
    def _create_basic_info_compare_table(self, model_infos: List[ModelInfoData]) -> PrettyTable:
        """创建模型基本信息比较表格"""
        table = PrettyTable()
        
        # 设置表头
        headers = ["属性"]
        for i, model_info in enumerate(model_infos, 1):
            headers.append(f"模型 {i}: {model_info.name}")
        
        table.field_names = headers
        table.title = "模型基本信息比较"
        
        # 添加基本信息比较
        basic_info = [
            ("模型名称", lambda info: info.name),
            ("模型类型", lambda info: info.model_type),
            ("框架", lambda info: info.framework),
            ("任务类型", lambda info: info.task_type),
            ("时间戳", lambda info: self._format_timestamp(info.timestamp)),
            ("指标数量", lambda info: len(info.metric_list)),
            ("标量指标", lambda info: len(info.get_metrics_by_type("scalar"))),
            ("曲线指标", lambda info: len(info.get_metrics_by_type("curve")))
        ]
        
        for label, getter in basic_info:
            row = [label]
            for model_info in model_infos:
                row.append(getter(model_info))
            table.add_row(row)
        
        # 美化表格
        for field in headers:
            table.align[field] = "l"
        
        return table
    
    def _create_scalar_metrics_compare_table(self, model_infos: List[ModelInfoData]) -> Optional[PrettyTable]:
        """创建标量指标比较表格"""
        # 收集所有唯一的标量指标名称
        all_scalar_metrics = set()
        for model_info in model_infos:
            scalar_metrics = model_info.get_metrics_by_type("scalar")
            all_scalar_metrics.update([m.name for m in scalar_metrics])
        
        if not all_scalar_metrics:
            return None
        
        table = PrettyTable()
        
        # 设置表头
        headers = ["指标名称"]
        for i, model_info in enumerate(model_infos, 1):
            headers.append(f"模型 {i}: {model_info.name}")
        
        table.field_names = headers
        table.title = "标量指标比较"
        
        # 添加标量指标比较数据
        for metric_name in sorted(all_scalar_metrics):
            row = [metric_name]
            
            # 获取每个模型的指标值
            for model_info in model_infos:
                metric = model_info.get_metric_by_name(metric_name)
                if metric:
                    value = metric.data.get("value", "-")
                    unit = metric.data.get("unit", "")
                    
                    # 格式化数值
                    if isinstance(value, float):
                        value = self._format_float_value(value)
                    
                    row.append(f"{value}{unit}")
                else:
                    row.append("- 无 -")
            
            table.add_row(row)
        
        # 美化表格
        for field in headers:
            if field != "指标名称":
                table.align[field] = "r"
            else:
                table.align[field] = "l"
        
        return table
    
    def _create_curve_key_values_compare_table(self, model_infos: List[ModelInfoData]) -> Optional[PrettyTable]:
        """创建曲线指标关键值比较表格"""
        # 收集所有唯一的曲线指标名称
        all_curve_metrics = set()
        for model_info in model_infos:
            curve_metrics = model_info.get_metrics_by_type("curve")
            all_curve_metrics.update([m.name for m in curve_metrics])
        
        if not all_curve_metrics:
            return None
        
        table = PrettyTable()
        
        # 设置表头
        headers = ["指标名称", "统计值"]
        for i, model_info in enumerate(model_infos, 1):
            headers.append(f"模型 {i}: {model_info.name}")
        
        table.field_names = headers
        table.title = "曲线指标关键值比较"
        
        # 添加曲线指标关键值比较数据
        for metric_name in sorted(all_curve_metrics):
            for stat_type in ["最大值", "最小值", "平均值"]:
                row = [metric_name, stat_type]
                
                for model_info in model_infos:
                    metric = model_info.get_metric_by_name(metric_name)
                    if metric:
                        values = metric.data.get("values", [])
                        valid_values = [v for v in values if v is not None]
                        
                        if valid_values:
                            if stat_type == "最大值":
                                value = max(valid_values)
                            elif stat_type == "最小值":
                                value = min(valid_values)
                            else:  # 平均值
                                value = sum(valid_values) / len(valid_values)
                            
                            # 格式化数值
                            if isinstance(value, float):
                                value = self._format_float_value(value)
                            
                            unit = metric.data.get("unit", "")
                            row.append(f"{value}{unit}")
                        else:
                            row.append("- 无 -")
                    else:
                        row.append("- 无 -")
                
                table.add_row(row)
            
            # 在每个指标后添加空行作为分隔
            if metric_name != list(sorted(all_curve_metrics))[-1]:
                table.add_row(["", ""] + ["" for _ in model_infos])
        
        # 美化表格
        for field in headers:
            if field not in ["指标名称", "统计值"]:
                table.align[field] = "r"
            else:
                table.align[field] = "l"
        
        return table
    
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