import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from prettytable import PrettyTable
from src.model_show_v2.data_models import ModelInfoData
from src.utils.log_utils.log_utils import get_logger
from .base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry

logger = get_logger(name=__name__)


@ModelVisualizerRegistry.register(namespace="models")
class ModelFileVisualizer(BaseModelVisualizer):
    """模型文件可视化器：可视化模型文件的信息"""

    def __init__(self):
        """初始化模型文件可视化器"""
        pass

    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        """判断是否为支持的模型文件信息

        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"

        Returns:
            bool: 是否支持该模型信息
        """
        # 验证路径格式是否符合模型文件规则
        if model_info.path:
            ext = os.path.splitext(model_info.path)[1].lower()
            return ext in ['.pth', '.pt', '.bin', '.onnx']
        
        return False

    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        """将模型文件信息可视化为表格格式

        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"

        Returns:
            Dict[str, Any]: 可视化结果，包含表格对象和显示文本
        """
        try:
            # 创建主表格
            table = PrettyTable()
            table.title = f"模型文件信息 ({os.path.basename(model_info.path)})"
            table.field_names = ["属性", "值"]
            
            # 添加基本信息
            table.add_row(["模型名称", model_info.name])
            table.add_row(["存储路径", model_info.path])
            table.add_row(["模型类型", model_info.model_type])
            table.add_row(["时间戳", self._format_timestamp(model_info.timestamp)])
            table.add_row(["框架", model_info.framework])
            table.add_row(["任务类型", model_info.task_type])
            table.add_row(["版本", model_info.version])
            
            # 添加分割线
            table.add_row(["="*20, "="*40])
            
            # 添加模型指标
            if model_info.metric_list:
                table.add_row(["模型指标", ""])
                
                for metric in model_info.metric_list:
                    value_str = f"{metric.data.get('value', '-')}{metric.data.get('unit', '')}"
                    table.add_row([f"  • {metric.name}", f"{value_str} ({metric.description})"])
            else:
                table.add_row(["📊  模型指标", "无指标数据"])
            
            # 添加分割线
            table.add_row(["="*20, "="*40])
            
            # 添加详细参数
            if model_info.params:
                table.add_row(["详细参数", ""])
                
                for param_name, param_value in model_info.params.items():
                    # 跳过已经作为指标显示的参数
                    if param_name not in ["parameters", "model_size"]:
                        table.add_row([f"  • {param_name}", self._format_value(param_value)])
            else:
                table.add_row(["⚙️  详细参数", "无参数数据"])
            
            # 美化表格
            table.align["属性"] = "l"
            table.align["值"] = "l"
            
            # 返回可视化结果
            return {
                "table": table,
                "text": str(table),
                "success": True,
                "message": "模型文件可视化成功"
            }
            
        except Exception as e:
            logger.error(f"模型文件可视化失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型文件可视化失败: {str(e)}"
            }
            
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        """比较多个模型文件的信息

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
                    "message": "比较需要至少2个模型文件信息"
                }
                
            # 创建比较表格
            table = PrettyTable()
            
            # 设置表头
            headers = ["模型属性"]
            for i, model_info in enumerate(model_infos, 1):
                headers.append(f"模型 {i}: {model_info.name}")
            
            table.field_names = headers
            
            # 添加基本信息比较
            basic_info = [
                ("模型名称", lambda info: info.name),
                ("存储路径", lambda info: os.path.basename(info.path)),
                ("模型类型", lambda info: info.model_type),
                ("时间戳", lambda info: self._format_timestamp(info.timestamp)),
                ("框架", lambda info: info.framework),
                ("任务类型", lambda info: info.task_type),
                ("版本", lambda info: info.version)
            ]
            
            for label, getter in basic_info:
                row = [label]
                for model_info in model_infos:
                    row.append(getter(model_info))
                table.add_row(row)
            
            # 添加分割线
            divider_row = ["="*20] + ["="*30 for _ in range(len(model_infos))]
            table.add_row(divider_row)
            
            # 收集所有唯一的指标名称
            all_metric_names = set()
            for model_info in model_infos:
                for metric in model_info.metric_list:
                    all_metric_names.add(metric.name)
            
            # 添加指标比较
            if all_metric_names:
                table.add_row(["模型指标", ""] + ["" for _ in range(len(model_infos) - 1)])
                
                for metric_name in sorted(all_metric_names):
                    row = [f"  • {metric_name}"]
                    for model_info in model_infos:
                        # 查找对应模型的指标
                        metric_value = "- 无 -"
                        for metric in model_info.metric_list:
                            if metric.name == metric_name:
                                metric_value = f"{metric.data.get('value', '-')}{metric.data.get('unit', '')}"
                                break
                        row.append(metric_value)
                    table.add_row(row)
            
            # 添加分割线
            table.add_row(divider_row)
            
            # 收集所有唯一的参数名称
            all_param_names = set()
            for model_info in model_infos:
                if model_info.params:
                    all_param_names.update(model_info.params.keys())
            
            # 跳过已经作为指标显示的参数
            param_names_to_compare = [name for name in all_param_names 
                                      if name not in ["parameters", "model_size"]]
            
            # 添加参数比较
            if param_names_to_compare:
                table.add_row(["模型参数", ""] + ["" for _ in range(len(model_infos) - 1)])
                
                for param_name in sorted(param_names_to_compare):
                    row = [f"  • {param_name}"]
                    for model_info in model_infos:
                        value = model_info.params.get(param_name, "- 无 -")
                        # 格式化复杂值
                        row.append(self._format_value(value))
                    table.add_row(row)
            
            # 美化表格
            for field in headers:
                table.align[field] = "l"
            
            # 返回比较结果
            return {
                "table": table,
                "text": str(table),
                "success": True,
                "message": "模型文件比较成功"
            }
            
        except Exception as e:
            logger.error(f"模型文件比较失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型文件比较失败: {str(e)}"
            }
            
    def _format_timestamp(self, timestamp: float) -> str:
        """格式化时间戳为可读日期时间

        Args:
            timestamp: 时间戳

        Returns:
            str: 格式化后的日期时间字符串
        """
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)
            
    def _format_value(self, value: Any) -> str:
        """格式化值为可读字符串

        Args:
            value: 任意类型的值

        Returns:
            str: 格式化后的字符串
        """
        if isinstance(value, (dict, list)):
            # 对于字典和列表，返回其字符串表示，但限制长度
            value_str = str(value)
            if len(value_str) > 50:
                return value_str[:50] + "..."
            return value_str
        elif value is None:
            return "- 无 -"
        else:
            return str(value)