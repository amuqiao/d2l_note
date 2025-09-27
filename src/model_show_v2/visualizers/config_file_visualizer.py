import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
from prettytable import PrettyTable
from src.model_show.data_models import ModelInfoData
from src.utils.log_utils import get_logger
from src.model_show.visualizers.base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry

logger = get_logger(name=__name__)


@ModelVisualizerRegistry.register(namespace="config")
class ConfigFileVisualizer(BaseModelVisualizer):
    """配置文件可视化器：可视化模型的配置信息"""

    def __init__(self):
        """初始化配置文件可视化器"""
        pass

    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        """判断是否为支持的配置信息

        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"

        Returns:
            bool: 是否支持该模型信息
        """
        # 严格匹配ConfigFileParser的支持规则
        # 1. 首先检查模型信息的命名空间是否为"config"
        # 2. 然后检查文件路径是否包含"config"且扩展名为.json
        if model_info.namespace != "config":
            return False
        
        # 验证路径格式是否符合config.json文件规则
        if model_info.path:
            ext = os.path.splitext(model_info.path)[1].lower()
            return ext == ".json" and "config" in model_info.path.lower()
        
        return False

    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        """将模型配置信息可视化为表格格式

        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"

        Returns:
            Dict[str, Any]: 可视化结果，包含表格对象和显示文本
        """
        try:
            # 创建主表格
            table = PrettyTable()
            table.title = f"模型配置信息 ({os.path.basename(model_info.path)})"
            table.field_names = ["属性", "值"]
            
            # 添加基本信息
            table.add_row(["模型名称", model_info.name])
            table.add_row(["存储路径", model_info.path])
            table.add_row(["模型类型", model_info.model_type])
            table.add_row(["时间戳", self._format_timestamp(model_info.timestamp)])
            table.add_row(["框架", model_info.framework])
            table.add_row(["任务类型", model_info.task_type])
            table.add_row(["版本", model_info.version])
            table.add_row(["命名空间", model_info.namespace])
            
            # 添加分割线
            table.add_row(["="*20, "="*40])
            
            # 添加详细配置参数
            if model_info.params:
                table.add_row(["详细配置", ""])  # 添加配置标题行
                
                # 分类显示不同类型的参数
                self._add_config_params_to_table(table, model_info.params)
            else:
                table.add_row(["⚙️  详细配置", "无配置参数"])
            
            # 美化表格
            table.align["属性"] = "l"
            table.align["值"] = "l"
            
            # 返回可视化结果
            return {
                "table": table,
                "text": str(table),
                "success": True,
                "message": "配置文件可视化成功"
            }
            
        except Exception as e:
            logger.error(f"配置文件可视化失败: {str(e)}")
            return {
                "success": False,
                "message": f"配置文件可视化失败: {str(e)}"
            }
            
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        """比较多个模型的配置信息

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
                    "message": "比较需要至少2个模型配置信息"
                }
                
            # 创建比较表格
            table = PrettyTable()
            
            # 设置表头
            headers = ["配置项"]
            for i, model_info in enumerate(model_infos, 1):
                headers.append(f"模型 {i}: {model_info.name}")
            
            table.field_names = headers
            
            # 收集所有唯一的配置项
            all_params = set()
            for model_info in model_infos:
                all_params.update(model_info.params.keys())
            
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
            
            # 添加配置参数比较
            if all_params:
                for param in sorted(all_params):
                    row = [f"{param}"]
                    for model_info in model_infos:
                        value = model_info.params.get(param, "- 无 -")
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
                "message": "模型配置比较成功"
            }
            
        except Exception as e:
            logger.error(f"模型配置比较失败: {str(e)}")
            return {
                "success": False,
                "message": f"模型配置比较失败: {str(e)}"
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
            # 对复杂数据类型进行格式化，避免表格过于冗长
            if isinstance(value, dict):
                if len(value) > 3:
                    return f"字典({len(value)}个键值对)"
                return json.dumps(value, ensure_ascii=False, indent=2)
            else:
                if len(value) > 5:
                    return f"列表({len(value)}个元素)"
                return json.dumps(value, ensure_ascii=False)
        return str(value)
        
    def _add_config_params_to_table(self, table: PrettyTable, params: Dict[str, Any]) -> None:
        """将配置参数添加到表格中

        Args:
            table: PrettyTable对象
            params: 配置参数字典
        """
        # 分类处理不同类型的参数
        model_params = {k: v for k, v in params.items() if k in ['model', 'architecture', 'network']}
        training_params = {k: v for k, v in params.items() if k in ['batch_size', 'epochs', 'lr', 'optimizer', 'loss']}
        data_params = {k: v for k, v in params.items() if k in ['dataset', 'data', 'input_size', 'output_size']}
        other_params = {k: v for k, v in params.items() if k not in list(model_params) + list(training_params) + list(data_params)}
        
        # 添加模型架构参数
        if model_params:
            table.add_row(["模型架构", ""])
            for key, value in model_params.items():
                table.add_row([f"  - {key}", self._format_value(value)])
        
        # 添加训练参数
        if training_params:
            table.add_row(["训练参数", ""])
            for key, value in training_params.items():
                table.add_row([f"  - {key}", self._format_value(value)])
        
        # 添加数据参数
        if data_params:
            table.add_row(["数据参数", ""])
            for key, value in data_params.items():
                table.add_row([f"  - {key}", self._format_value(value)])
        
        # 添加其他参数
        if other_params:
            table.add_row(["其他参数", ""])
            for key, value in other_params.items():
                table.add_row([f"  - {key}", self._format_value(value)])


if __name__ == "__main__":
    """用于单独验证visualize和compare方法的输出"""
    # 创建模拟的ModelInfoData对象
    def create_mock_model_info(config_path: str, name: str, params: Dict[str, Any]) -> ModelInfoData:
        return ModelInfoData(
            name=name,
            path=config_path,
            model_type="test_model",
            timestamp=datetime.now().timestamp(),
            framework="test_framework",
            task_type="test_task",
            version="1.0.0",
            namespace="config",
            params=params
        )
    
    # 创建第一个模型配置
    params1 = {
        "model": {
            "type": "CNN",
            "layers": 5
        },
        "batch_size": 32,
        "epochs": 100,
        "lr": 0.001,
        "optimizer": "Adam",
        "dataset": "MNIST",
        "input_size": [28, 28, 1]
    }
    model_info1 = create_mock_model_info(
        config_path="/path/to/config1.json", 
        name="model_v1", 
        params=params1
    )
    
    # 创建第二个模型配置
    params2 = {
        "model": {
            "type": "CNN",
            "layers": 10
        },
        "batch_size": 64,
        "epochs": 200,
        "lr": 0.0001,
        "optimizer": "SGD",
        "dataset": "MNIST",
        "output_size": 10,
        "dropout": 0.5
    }
    model_info2 = create_mock_model_info(
        config_path="/path/to/config2.json", 
        name="model_v2", 
        params=params2
    )
    
    # 初始化可视化器
    visualizer = ConfigFileVisualizer()
    
    print("=" * 80)
    print("测试可视化单个模型配置")
    print("=" * 80)
    result1 = visualizer.visualize(model_info1)
    if result1["success"]:
        print(result1["text"])
    else:
        print(f"错误: {result1['message']}")
    
    print("\n" + "=" * 80)
    print("测试比较多个模型配置")
    print("=" * 80)
    compare_result = visualizer.compare([model_info1, model_info2])
    if compare_result["success"]:
        print(compare_result["text"])
    else:
        print(f"错误: {compare_result['message']}")
