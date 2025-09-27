from typing import Optional, Dict, List
import os
import json
import re
from datetime import datetime
from .base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData, MetricData


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
            
            # 从文件路径中提取模型名称和时间戳
            file_dir = os.path.dirname(file_path)
            model_name = self._extract_model_name(file_dir)
            timestamp = os.path.getmtime(file_path)
            
            # 创建ModelInfoData对象
            model_info = ModelInfoData(
                name=model_name,
                path=file_dir,  # 使用目录作为path，而不是单个文件
                model_type=self._infer_model_type(file_dir),
                timestamp=timestamp,
                params={},
                framework="PyTorch",  # 假设为PyTorch模型
                task_type="classification",  # 基于指标判断为分类任务
                version="1.0"
            )
            
            # 添加标量指标
            self._add_scalar_metrics(model_info, metrics_data, file_path, timestamp)
            
            # 添加曲线指标
            if 'epoch_metrics' in metrics_data:
                # 过滤掉best_test_acc，因为它应该只作为scalar类型
                curve_metrics_data = {}
                for key, value in metrics_data.items():
                    if key != 'best_test_acc':
                        curve_metrics_data[key] = value
                self._add_curve_metrics(model_info, curve_metrics_data['epoch_metrics'], file_path, timestamp)
            
            # 添加训练时间信息到params
            if 'total_training_time' in metrics_data:
                model_info.params['total_training_time'] = metrics_data['total_training_time']
            
            if 'samples_per_second' in metrics_data:
                model_info.params['samples_per_second'] = metrics_data['samples_per_second']
            
            # 添加训练时间范围
            if 'training_start_time' in metrics_data:
                model_info.params['training_start_time'] = metrics_data['training_start_time']
            if 'training_end_time' in metrics_data:
                model_info.params['training_end_time'] = metrics_data['training_end_time']
            
            return model_info
            
        except Exception as e:
            raise ValueError(f"解析指标文件失败: {str(e)}")
    
    def _extract_model_name(self, directory_path: str) -> str:
        """从目录路径提取模型名称"""
        if not directory_path:
            return 'unknown_model'
        
        # 尝试从目录名中提取模型名称（如run_20250914_040635可能对应LeNet模型）
        dir_name = os.path.basename(directory_path)
        
        # 检查目录中是否有模型文件，从中提取模型名称
        for file in os.listdir(directory_path):
            if file.endswith('.pth') and 'best_model' in file.lower():
                # 从文件名提取模型名称，如best_model_LeNet_acc_0.7860_epoch_9.pth
                match = re.search(r'best_model_([A-Za-z0-9_]+)_acc', file)
                if match:
                    return match.group(1)
        
        return dir_name
    
    def _infer_model_type(self, directory_path: str) -> str:
        """推断模型类型"""
        # 尝试从目录或文件名推断模型类型
        for file in os.listdir(directory_path):
            if file.endswith('.pth'):
                if 'lenet' in file.lower():
                    return 'CNN-LeNet'
                elif 'alexnet' in file.lower():
                    return 'CNN-AlexNet'
                elif 'resnet' in file.lower():
                    return 'CNN-ResNet'
                elif 'vgg' in file.lower():
                    return 'CNN-VGG'
                elif 'vit' in file.lower():
                    return 'Transformer-ViT'
        
        return 'unknown'
    
    def _add_scalar_metrics(self, model_info: ModelInfoData, metrics_data: Dict, 
                          file_path: str, timestamp: float) -> None:
        """添加标量类型的指标"""
        # 定义需要添加的标量指标
        scalar_metrics = [
            ('final_train_loss', 'Final Training Loss', 0.0, ''),
            ('final_train_acc', 'Final Training Accuracy', 0.0, '%'),
            ('final_test_acc', 'Final Test Accuracy', 0.0, '%'),
            ('best_test_acc', 'Best Test Accuracy', 0.0, '%')
        ]
        
        for key, name, default, unit in scalar_metrics:
            value = metrics_data.get(key, default)
            # 如果是准确率，转换为百分比
            if unit == '%' and isinstance(value, (int, float)):
                value = value * 100
            
            metric_data = MetricData(
                name=name,
                metric_type="scalar",
                data={"value": value, "unit": unit},
                source_path=file_path,
                timestamp=timestamp,
                description=f"{name} of the model"
            )
            model_info.add_metric(metric_data)
    
    def _add_curve_metrics(self, model_info: ModelInfoData, epoch_metrics: List[Dict],
                          file_path: str, timestamp: float) -> None:
        """添加曲线类型的指标"""
        # 定义需要添加的曲线指标
        curve_metrics = [
            ('train_loss', 'Training Loss', False),
            ('train_acc', 'Training Accuracy', True),
            ('test_acc', 'Test Accuracy', True),
            ('epoch_time', 'Epoch Time', False)
        ]
        
        # 提取epochs列表
        epochs = [item['epoch'] for item in epoch_metrics]
        
        for key, name, is_percentage in curve_metrics:
            # 提取每个指标的值
            values = []
            for item in epoch_metrics:
                if key in item:
                    value = item[key]
                    # 如果是准确率，转换为百分比
                    if is_percentage and isinstance(value, (int, float)):
                        value = value * 100
                    values.append(value)
                else:
                    values.append(None)  # 处理缺失值
            
            unit = '%' if is_percentage else ''
            unit = 's' if key == 'epoch_time' else unit
            
            metric_data = MetricData(
                name=name,
                metric_type="curve",
                data={"epochs": epochs, "values": values, "unit": unit},
                source_path=file_path,
                timestamp=timestamp,
                description=f"{name} across training epochs"
            )
            model_info.add_metric(metric_data)
