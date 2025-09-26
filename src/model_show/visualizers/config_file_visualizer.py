import json
import os
from typing import Dict, Any
from src.model_show.data_models import ModelInfoData
from src.utils.log_utils import get_logger
from .base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry

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
        """将配置文件信息可视化为结构化数据

        Args:
            model_info: 模型信息数据
            namespace: 命名空间，默认为"default"

        Returns:
            Dict[str, Any]: 可视化结果
        """
        try:
            # 构建可视化结果
            visualization_result = {
                "type": "config_visualization",
                "model_name": model_info.name,
                "model_path": model_info.path,
                "model_type": model_info.model_type,
                "framework": model_info.framework,
                "task_type": model_info.task_type,
                "timestamp": model_info.timestamp,
                "version": model_info.version,
                "namespace": model_info.namespace,
                "config_overview": self._generate_config_overview(model_info.params),
                "config_details": self._format_config_details(model_info.params),
                "visualization_elements": self._create_visualization_elements(model_info.params)
            }

            return visualization_result

        except Exception as e:
            logger.error(f"可视化配置文件失败 {model_info.name}: {str(e)}")
            return {
                "type": "error",
                "error_message": str(e),
                "model_name": model_info.name
            }

    def _generate_config_overview(self, config_params: Dict[str, any]) -> Dict[str, any]:
        """生成配置概览信息

        Args:
            config_params: 配置参数字典

        Returns:
            Dict[str, any]: 配置概览
        """
        overview = {
            "parameter_count": len(config_params),
            "has_model_config": "model_config" in config_params,
            "has_training_config": "training_config" in config_params,
            "has_optimizer_config": "optimizer" in config_params or "optimizer_config" in config_params,
            "has_dataset_config": "dataset" in config_params or "dataset_config" in config_params
        }

        # 添加关键参数信息
        key_params = []
        for key, value in config_params.items():
            if isinstance(value, (int, float, str)) and len(key_params) < 10:
                key_params.append({"name": key, "value": value})
        overview["key_parameters"] = key_params

        return overview

    def _format_config_details(self, config_params: Dict[str, any]) -> Dict[str, any]:
        """格式化配置详情

        Args:
            config_params: 配置参数字典

        Returns:
            Dict[str, any]: 格式化后的配置详情
        """
        # 这里可以根据需要对配置参数进行格式化，使其更适合显示
        # 为了简单起见，我们直接返回处理后的配置参数
        formatted_config = {}
        
        for key, value in config_params.items():
            # 将复杂对象转换为字符串，便于显示
            if isinstance(value, (dict, list)):
                try:
                    formatted_config[key] = json.dumps(value, ensure_ascii=False, indent=2)
                except Exception:
                    formatted_config[key] = str(value)
            else:
                formatted_config[key] = value
                
        return formatted_config

    def _create_visualization_elements(self, config_params: Dict[str, any]) -> Dict[str, any]:
        """创建可视化元素

        Args:
            config_params: 配置参数字典

        Returns:
            Dict[str, any]: 可视化元素定义
        """
        # 生成不同类型的可视化元素
        elements = {
            "config_tree": {
                "type": "tree",
                "data": self._build_config_tree(config_params)
            },
            "param_distribution": {
                "type": "chart",
                "chart_type": "pie",
                "title": "配置参数类型分布",
                "data": self._generate_param_distribution(config_params)
            }
        }

        return elements

    def _build_config_tree(self, config_params: Dict[str, any]) -> Dict[str, any]:
        """构建配置树状结构

        Args:
            config_params: 配置参数字典

        Returns:
            Dict[str, any]: 树状结构数据
        """
        tree = {"name": "config", "children": []}
        
        for key, value in config_params.items():
            node = {"name": key}
            if isinstance(value, dict):
                node["children"] = self._dict_to_tree_children(value)
            elif isinstance(value, list):
                node["children"] = self._list_to_tree_children(value)
            else:
                node["value"] = str(value)
            
            tree["children"].append(node)
            
        return tree

    def _dict_to_tree_children(self, data: Dict[str, any]) -> list:
        """将字典转换为树状子节点列表

        Args:
            data: 字典数据

        Returns:
            list: 子节点列表
        """
        children = []
        for key, value in data.items():
            child = {"name": key}
            if isinstance(value, dict):
                child["children"] = self._dict_to_tree_children(value)
            elif isinstance(value, list):
                child["children"] = self._list_to_tree_children(value)
            else:
                child["value"] = str(value)
            children.append(child)
        return children

    def _list_to_tree_children(self, data: list) -> list:
        """将列表转换为树状子节点列表

        Args:
            data: 列表数据

        Returns:
            list: 子节点列表
        """
        children = []
        for i, item in enumerate(data):
            child = {"name": f"[{i}]", "index": i}
            if isinstance(item, dict):
                child["children"] = self._dict_to_tree_children(item)
            elif isinstance(item, list):
                child["children"] = self._list_to_tree_children(item)
            else:
                child["value"] = str(item)
            children.append(child)
        return children

    def _generate_param_distribution(self, config_params: Dict[str, any]) -> list:
        """生成参数类型分布数据

        Args:
            config_params: 配置参数字典

        Returns:
            list: 分布数据列表
        """
        type_count = {}
        
        for value in config_params.values():
            value_type = type(value).__name__
            if value_type not in type_count:
                type_count[value_type] = 0
            type_count[value_type] += 1
        
        return [
            {"name": key, "value": value} for key, value in type_count.items()
        ]