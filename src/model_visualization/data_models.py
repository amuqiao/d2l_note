from typing import Dict, Any
from dataclasses import dataclass


# ==================================================
# 核心数据模型层：标准化数据表示，增强核心数据模型地位
# ==================================================
@dataclass
class MetricData:
    """标准化指标数据模型：统一不同指标文件的内存表示"""

    metric_type: str  # 指标类型："epoch_curve" / "confusion_matrix" / "lr_curve" 等
    data: Dict[str, Any]  # 结构化指标数据
    source_path: str  # 数据来源文件路径
    timestamp: float  # 数据生成时间戳


@dataclass
class ModelInfoData:
    """模型信息数据模型：统一模型和训练任务的信息表示"""

    type: str  # "run" 或 "model"
    path: str  # 路径（目录或文件）
    model_type: str  # 模型类型
    params: Dict[str, Any]  # 模型参数
    metrics: Dict[str, Any]  # 性能指标
    timestamp: float  # 时间戳
    namespace: str = "default"  # 命名空间，默认为"default"
