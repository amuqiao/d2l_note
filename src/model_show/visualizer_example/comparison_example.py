"""
模型比较可视化器示例：展示如何实现compare方法

这个示例展示了如何创建一个支持模型比较功能的可视化器，
实现BaseModelVisualizer接口中定义的compare抽象方法。
"""
from typing import Dict, List, Any
from src.model_show.data_models import ModelInfoData
from src.model_show.visualizers.base_model_visualizers import BaseModelVisualizer, ModelVisualizerRegistry
from src.utils.log_utils import get_logger

logger = get_logger(name=__name__)


@ModelVisualizerRegistry.register(namespace="example")
class ExampleComparisonVisualizer(BaseModelVisualizer):
    """示例比较可视化器：演示如何实现compare方法"""
    
    def __init__(self):
        """初始化示例比较可视化器"""
        # 设置较高优先级，确保在示例中被优先选择
        self.priority = 100
    
    def support(self, model_info: ModelInfoData, namespace: str = "default") -> bool:
        """判断是否支持该模型信息
        
        这个示例可视化器支持所有类型的ModelInfoData
        """
        return True
    
    def visualize(self, model_info: ModelInfoData, namespace: str = "default") -> Dict[str, Any]:
        """将单个模型信息可视化为指定格式"""
        return {
            "type": "single_model_visualization",
            "model_name": model_info.name,
            "model_path": model_info.path,
            "model_type": model_info.model_type,
            "params_count": len(model_info.params),
            "metrics_count": len(model_info.metrics),
            "timestamp": model_info.timestamp,
            "namespace": model_info.namespace
        }
    
    def compare(self, model_infos: List[ModelInfoData], namespace: str = "default") -> Dict[str, Any]:
        """比较多个模型信息并可视化为指定格式"""
        # 收集所有模型的基本信息
        models_basic_info = []
        
        # 收集所有唯一的指标名称
        all_metric_names = set()
        
        for info in model_infos:
            # 收集模型基本信息
            model_info_dict = {
                "name": info.name,
                "path": info.path,
                "type": info.model_type,
                "namespace": info.namespace,
                "timestamp": info.timestamp,
                "params_count": len(info.params),
                "metrics": {}
            }
            
            # 收集指标信息
            for metric_name, metric_value in info.metrics.items():
                model_info_dict["metrics"][metric_name] = metric_value
                all_metric_names.add(metric_name)
            
            models_basic_info.append(model_info_dict)
        
        # 计算指标统计信息
        metrics_stats = {}
        for metric_name in all_metric_names:
            # 收集所有模型的该指标值（只考虑数值类型）
            metric_values = []
            for info in model_infos:
                if (metric_name in info.metrics and 
                    isinstance(info.metrics[metric_name], (int, float))):
                    metric_values.append(info.metrics[metric_name])
            
            # 计算统计信息（如果有足够的数据）
            if metric_values:
                metrics_stats[metric_name] = {
                    "count": len(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "average": sum(metric_values) / len(metric_values)
                }
        
        # 构建比较结果
        comparison_result = {
            "type": "models_comparison",
            "models_count": len(model_infos),
            "models_info": models_basic_info,
            "common_metrics": list(all_metric_names),
            "metrics_statistics": metrics_stats,
            "comparison_timestamp": ModelInfoData.timestamp if hasattr(ModelInfoData, 'timestamp') else None
        }
        
        return comparison_result


# 使用示例
if __name__ == "__main__":
    # 创建一些示例ModelInfoData对象
    from datetime import datetime
    
    # 模型1
    model1 = ModelInfoData(
        name="model_a",
        path="models/model_a.pth",
        model_type="CNN",
        timestamp=datetime.now().timestamp(),
        params={"layers": 10, "filters": 64},
        metrics={"accuracy": 0.85, "loss": 0.32},
        framework="PyTorch",
        task_type="classification"
    )
    
    # 模型2
    model2 = ModelInfoData(
        name="model_b",
        path="models/model_b.pth",
        model_type="Transformer",
        timestamp=datetime.now().timestamp(),
        params={"layers": 6, "heads": 8},
        metrics={"accuracy": 0.89, "loss": 0.28},
        framework="PyTorch",
        task_type="classification"
    )
    
    # 测试单个模型可视化
    print("\n=== 单个模型可视化 ===")
    from src.model_show.visualizer_registry import visualize_model_info
    single_result = visualize_model_info(model1, namespace="example")
    print(f"可视化结果: {single_result}")
    
    # 测试多个模型比较
    print("\n=== 多个模型比较 ===")
    from src.model_show.visualizer_registry import compare_model_infos
    comparison_result = compare_model_infos([model1, model2], namespace="example")
    print(f"比较结果: {comparison_result}")
    
    # 直接通过注册中心调用
    print("\n=== 通过注册中心调用 ===")
    registry_result = ModelVisualizerRegistry.compare_models([model1, model2], namespace="example")
    print(f"注册中心比较结果: {registry_result}")