from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime


# 第一步：保留独立的MetricData（确保指标信息结构化）
@dataclass
class MetricData:
    """标准化指标数据模型：存储单个指标的详细信息（标量/曲线/矩阵等）"""
    # 核心标识：确保能唯一区分指标
    name: str  # 指标名称（如"train_loss"、"val_accuracy"、"confusion_matrix"）
    metric_type: str  # 指标类型（严格限定，便于可视化器匹配）：
                      # - "scalar"：单个数值（如最终准确率95%）
                      # - "curve"：时序曲线（如每epoch的loss）
                      # - "matrix"：矩阵数据（如混淆矩阵）
                      # - "image"：图像类指标（如特征图、错误案例）
    
    # 指标核心数据：根据metric_type有不同结构
    data: Dict[str, Any]  # 结构化数据示例：
                          # - scalar：{"value": 0.95, "unit": "%"}
                          # - curve：{"epochs": [1,2,...,100], "values": [0.8, 0.85, ..., 0.95]}
                          # - matrix：{"classes": ["cat", "dog"], "data": [[98,2],[5,95]]}
    
    # 元数据：追溯指标来源
    source_path: str  # 指标文件路径（如"./models/resnet50/metrics/val_acc.json"）
    timestamp: float  # 指标生成时间戳（如文件修改时间、训练完成时间）
    
    # 可选扩展：满足个性化需求
    description: str = ""  # 指标描述（如"验证集准确率，每epoch计算一次"）
    tags: Dict[str, str] = field(default_factory=dict)  # 分类标签（如{"stage": "val", "frequency": "epoch"}）
    extra: Dict[str, Any] = field(default_factory=dict)  # 自定义字段（如"best_epoch": 80）


# 第二步：合并后的ModelInfoData（包含MetricData列表）
@dataclass
class ModelInfoData:
    """核心模型信息数据模型：整合模型基本信息、参数配置、所有关联指标"""
    # ------------------------------
    # 1. 模型核心标识（必选，唯一确定一个模型）
    # ------------------------------
    name: str  # 模型名称（如"resnet50_v1"、"vit_base_patch16"）
    path: str  # 模型存储根路径（而非单个文件，便于关联多个指标文件）
               # 示例："./models/resnet50/"（该目录下包含模型文件、配置文件、metrics子目录）
    model_type: str  # 模型结构类型（如"CNN"、"Transformer"、"RNN"、"GAN"）
    timestamp: float  # 模型最终生成时间戳（如训练完成时间、模型导出时间）
    
    # ------------------------------
    # 2. 模型核心属性（可选，按需求填充）
    # ------------------------------
    framework: str = ""  # 训练框架（如"PyTorch_2.1"、"TensorFlow_2.15"，避免空值用默认）
    task_type: str = ""  # 任务类型（如"image_classification"、"object_detection"、"nlp_sentiment"）
    params: Dict[str, Any] = field(default_factory=dict)  # 模型参数配置（结构化存储）
                                                          # 示例：{"input_size": [3,224,224], "num_classes": 1000, "total_params": 25600000}
    description: str = ""  # 模型描述（如"基于ResNet50的图像分类模型，在ImageNet上预训练"）
    version: str = "1.0"  # 模型版本（便于迭代管理，如"1.0"为基础版，"1.1"为优化版）
    
    # ------------------------------
    # 3. 模型关联指标（核心新增：MetricData列表）
    # ------------------------------
    metric_list: List[MetricData] = field(default_factory=list)  # 该模型的所有指标
    
    # ------------------------------
    # 4. 扩展字段（兼容未来需求）
    # ------------------------------
    extra: Dict[str, Any] = field(default_factory=dict)  # 自定义信息（如"train_duration": "24h", "gpu_type": "A100"）
    
    # ------------------------------
    # 5. 指标管理辅助方法（简化指标查找/新增/删除）
    # ------------------------------
    def add_metric(self, metric: MetricData) -> None:
        """新增指标：自动去重（避免重复添加同名指标）"""
        existing_metric = self.get_metric_by_name(metric.name)
        if existing_metric:
            # 可选：覆盖现有指标（或抛警告，根据业务需求调整）
            self.metric_list = [m for m in self.metric_list if m.name != metric.name]
        self.metric_list.append(metric)
    
    def get_metric_by_name(self, metric_name: str) -> Optional[MetricData]:
        """按指标名称查找：快速定位单个指标（如查找"val_accuracy"）"""
        for metric in self.metric_list:
            if metric.name == metric_name:
                return metric
        return None
    
    def get_metrics_by_type(self, metric_type: str) -> List[MetricData]:
        """按指标类型筛选：批量获取同类指标（如获取所有"curve"类型指标，用于绘制曲线）"""
        return [metric for metric in self.metric_list if metric.metric_type == metric_type]
    
    def remove_metric(self, metric_name: str) -> bool:
        """删除指标：返回是否删除成功"""
        initial_count = len(self.metric_list)
        self.metric_list = [m for m in self.metric_list if m.name != metric_name]
        return len(self.metric_list) < initial_count