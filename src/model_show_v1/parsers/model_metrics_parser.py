from typing import Optional
import os
import json
from datetime import datetime
from .base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show.data_models import ModelInfoData


@ModelInfoParserRegistry.register(namespace="metrics")
class MetricsFileParser(BaseModelInfoParser):
    """指标文件解析器：解析模型的性能指标文件"""
    
    # 设置较高优先级
    priority: int = 70
    
    def support(self, file_path: str, namespace: str = "default") -> bool:
        """判断是否为支持的指标文件格式
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            bool: 是否支持该文件
        """
        if not os.path.exists(file_path):
            return False
            
        # 支持.json格式的指标文件
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.json' and 'metrics' in os.path.basename(file_path).lower()
    
    def parse(self, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        """解析指标文件为ModelInfoData对象
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            ModelInfoData: 解析后的模型信息数据
        """
        try:
            with open(file_path, 'r') as f:
                metrics_data = json.load(f)
            
            # 从文件路径中提取模型名称
            file_dir = os.path.dirname(file_path)
            model_name = os.path.basename(file_dir) if file_dir else 'unknown_model'
            
            # 提取关键指标信息
            metrics = {
                'final_train_loss': metrics_data.get('final_train_loss', 0.0),
                'final_train_acc': metrics_data.get('final_train_acc', 0.0),
                'final_test_acc': metrics_data.get('final_test_acc', 0.0),
                'best_test_acc': metrics_data.get('best_test_acc', 0.0),
                'total_training_time': metrics_data.get('total_training_time', '0s'),
                'samples_per_second': metrics_data.get('samples_per_second', '0')
            }
            
            # 提取时间戳信息
            timestamp = os.path.getmtime(file_path)
            
            # 创建并返回ModelInfoData对象
            return ModelInfoData(
                name=model_name,
                path=file_path,
                model_type="unknown",  # 从指标文件无法直接获取模型类型
                timestamp=timestamp,
                params={},
                metrics=metrics,
                framework="PyTorch",  # 假设为PyTorch模型
                task_type="classification",  # 基于指标判断为分类任务
                version="1.0"
            )
            
        except Exception as e:
            raise ValueError(f"解析指标文件失败: {str(e)}")
