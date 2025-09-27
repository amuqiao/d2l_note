from typing import Dict, Any
from dataclasses import dataclass


# ==================================================
# 核心数据模型层：标准化数据表示，增强核心数据模型地位
# ==================================================

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricData:
    """标准化指标数据模型：统一不同指标文件的内存表示"""

    # 核心标识信息
    name: str  # 指标名称（如"accuracy"、"loss"）
    data: Dict[str, Any]  # 结构化指标数据

    # 元数据信息
    metric_type: str  # 指标类型："scalar" / "curve" / "matrix" / "image" 等
    source_path: str  # 数据来源文件路径
    timestamp: float  # 数据生成时间戳

    # 可选扩展信息
    namespace: str = "default"
    version: str = "1.0"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    source_name: str = ""  # 数据来源名称（如训练任务名）
    source_type: str = "file"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfoData:
    """模型信息数据模型：统一模型和训练任务的信息表示"""

    # 核心标识信息
    name: str  # 模型名称
    path: str  # 模型存储路径
    model_type: str  # 模型类型（如"CNN"、"Transformer"）
    timestamp: float  # 创建/训练完成时间戳

    # 核心属性信息
    params: Dict[str, Any] = field(default_factory=dict)  # 模型参数配置
    metrics: Dict[str, Any] = field(default_factory=dict)  # 关键性能指标

    # 可选扩展信息
    namespace: str = "default"
    description: str = ""  # 模型描述
    framework: str = ""  # 框架（如"PyTorch"、"TensorFlow"）
    task_type: str = ""  # 任务类型（如"classification"、"detection"）
    version: str = "1.0"  # 模型版本
    extra: Dict[str, Any] = field(default_factory=dict)
