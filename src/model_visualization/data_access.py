import os
import json
import torch
from typing import Optional, Any

# 导入自定义日志模块
from src.utils.log_utils.log_utils import get_logger

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/data_access.log", global_level="INFO")

# ==================================================
# 数据访问层：负责数据的读取和解析，提供统一的访问接口
# ==================================================
class DataAccessor:
    """数据访问器：提供统一的数据读取和解析接口"""
    
    @staticmethod
    def read_file(file_path: str) -> Optional[Any]:
        """通用文件读取方法"""
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.pth'):
                # 检查PyTorch版本是否支持weights_only参数
                try:
                    return torch.load(file_path, map_location="cpu", weights_only=True)
                except TypeError:
                    return torch.load(file_path, map_location="cpu")
            else:
                logger.warning(f"不支持的文件类型: {file_path}")
                return None
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None

    @staticmethod
    def get_file_timestamp(file_path: str) -> float:
        """获取文件的修改时间戳"""
        if os.path.exists(file_path):
            return os.path.getmtime(file_path)
        return 0.0