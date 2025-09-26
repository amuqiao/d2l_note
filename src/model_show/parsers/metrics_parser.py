from typing import Optional
import os
import json
from .base_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_visualization.data_models import ModelInfoData


@ModelInfoParserRegistry.register(namespace="metrics")
class MetricsFileParser(BaseModelInfoParser):
    """指标文件解析器：解析模型的性能指标文件"""
    
    # 设置较高优先级
    priority: int = 70
    
    def support(self, file_path: str) -> bool:
        """判断是否为支持的指标文件格式"""
        if not os.path.exists(file_path):
            return False
            
        # 支持.json格式的指标文件
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.json' and 'metrics' in os.path.basename(file_path).lower()
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        """解析指标文件为ModelInfoData对象"""
        try:
            with open(file_path, 'r') as f:
                metrics_data = json.load(f)
            
            # 提取模型指标信息
            accuracy = metrics_data.get('accuracy', 0.0)
            loss = metrics_data.get('loss', 0.0)
            model_name = metrics_data.get('model_name', 'unknown')
            
            # 创建并返回ModelInfoData对象
            return ModelInfoData(
                file_path=file_path,
                model_name=model_name,
                accuracy=accuracy,
                loss=loss,
                # 其他字段根据实际情况填充
            )
            
        except Exception as e:
            raise ValueError(f"解析指标文件失败: {str(e)}")
