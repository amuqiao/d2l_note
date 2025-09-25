from typing import Optional
import os
import torch
from abc import ABC, abstractmethod
from src.model_visualization.data_models import ModelInfoData
from src.model_visualization.data_access import DataAccessor
from src.utils.log_utils import get_logger
from src.model_visualization.model_info_parsers import BaseModelInfoParser
from .timestride_training_metrics_parser import TimestrideTrainingMetricsParser

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_parser.log", global_level="INFO")


class TimestrideModelFileParser(BaseModelInfoParser):
    """Timestride模型文件解析器：处理.pth模型文件"""
    
    def support(self, file_path: str) -> bool:
        return "timestride" in file_path and file_path.endswith(".pth")
    
    def parse(self, file_path: str) -> Optional[ModelInfoData]:
        try:
            model_data = DataAccessor.read_file(file_path)
            if not model_data:
                return None
            
            # 确定模型类型
            model_type = "Unknown"
            if isinstance(model_data, dict):
                # 检查模型是否包含state_dict或model_state_dict
                if "model_state_dict" in model_data:
                    state_dict = model_data["model_state_dict"]
                elif "state_dict" in model_data:
                    state_dict = model_data["state_dict"]
                else:
                    state_dict = model_data
                
                # 从state_dict中分析模型结构特征来推断模型类型
                if any(key.startswith("conv") for key in state_dict.keys()):
                    if any(key.startswith("fc") for key in state_dict.keys()):
                        model_type = "CNN with FC layers"
                    else:
                        model_type = "CNN"
                elif any(key.startswith("linear") for key in state_dict.keys()):
                    model_type = "MLP"
                elif any(key.startswith("times_blocks") for key in state_dict.keys()):
                    model_type = "TimesNet"
                
                # 提取参数量信息
                total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            else:
                total_params = 0
                state_dict = {}
            
            # 从文件名中提取更多信息
            file_name = os.path.basename(file_path)
            
            # 构建模型参数
            params = {
                "file_name": file_name,
                "total_params": total_params,
                "has_state_dict": isinstance(model_data, dict) and ("state_dict" in model_data or "model_state_dict" in model_data)
            }
            
            # 尝试从同目录下的args.json获取更多信息
            run_dir = os.path.dirname(file_path)
            args_path = os.path.join(run_dir, "args.json")
            if os.path.exists(args_path):
                args_data = DataAccessor.read_file(args_path)
                if args_data:
                    # 合并配置文件的参数
                    args_params = {
                        "model_name": args_data.get("model", "Unknown"),
                        "task_name": args_data.get("task_name", "Unknown"),
                        "batch_size": args_data.get("batch_size", 0),
                        "learning_rate": args_data.get("learning_rate", 0.0),
                        "device": args_data.get("device", "cpu")
                    }
                    params.update(args_params)
                    if args_params["model_name"] != "Unknown":
                        model_type = args_params["model_name"]
            
            # 指标信息（可能不完整）
            metrics = {}
            
            # 尝试从同目录下的training_metrics.json获取性能指标
            metrics_path = os.path.join(run_dir, "training_metrics.json")
            if os.path.exists(metrics_path):
                metrics_parser = TimestrideTrainingMetricsParser()
                metrics_data = metrics_parser.parse(metrics_path)
                if metrics_data:
                    metrics = metrics_data.metrics
            
            timestamp = DataAccessor.get_file_timestamp(file_path)
            
            return ModelInfoData(
                type="model",
                path=file_path,
                model_type=model_type,
                params=params,
                metrics=metrics,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"解析Timestride模型文件失败 {file_path}: {str(e)}")
            return None


# 导入基类（需要放在文件末尾以避免循环导入）
from src.model_visualization.model_info_parsers import BaseModelInfoParser
