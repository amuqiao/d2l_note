from typing import Optional
import os
import torch
from .base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData, MetricData
from src.utils.log_utils.log_utils import get_logger

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_parser.log", global_level="INFO")


@ModelInfoParserRegistry.register(namespace="models")
class ModelFileParser(BaseModelInfoParser):
    """模型文件解析器：解析PyTorch等模型文件，提取模型结构和参数信息"""
    
    # 设置较高优先级，确保模型文件被正确解析
    priority: int = 80
    
    def support(self, file_path: str, namespace: str = "default") -> bool:
        """判断是否为支持的模型文件格式
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            bool: 是否支持该文件
        """
        if not os.path.exists(file_path):
            return False
            
        # 支持PyTorch模型文件和ONNX模型文件
        ext = os.path.splitext(file_path)[1].lower()
        return ext in ['.pth', '.pt', '.bin', '.onnx']
    
    def parse(self, file_path: str, namespace: str = "default") -> Optional[ModelInfoData]:
        """解析模型文件为ModelInfoData对象
        
        Args:
            file_path: 文件路径
            namespace: 命名空间，默认为"default"
        
        Returns:
            ModelInfoData: 解析后的模型信息数据
        """
        try:
            # 提取文件名作为默认模型名
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 尝试加载模型以获取更多信息
            model_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
            model_params = None
            input_shape = None
            model_type = "unknown"
            
            # 根据文件扩展名尝试不同的加载方式
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in ['.pth', '.pt']:
                model_type = "PyTorch"
                # 尝试加载模型（不加载权重以节省内存）
                try:
                    # 对于保存的完整模型
                    model = torch.load(file_path, map_location=torch.device('cpu'))
                    
                    # 计算参数数量
                    if hasattr(model, 'parameters'):
                        model_params = sum(p.numel() for p in model.parameters())
                    
                    # 尝试获取输入形状（这取决于模型是否有相关信息）
                    if hasattr(model, 'input_shape'):
                        input_shape = model.input_shape
                        
                except Exception as e:
                    logger.warning(f"无法完全加载PyTorch模型 {file_path}: {str(e)}")
            
            elif ext == '.onnx':
                model_type = "ONNX"
                # ONNX模型解析可以使用onnx库
                try:
                    import onnx
                    onnx_model = onnx.load(file_path)
                    # 获取输入形状信息
                    if onnx_model.graph.input:
                        input_shape = []
                        for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim:
                            input_shape.append(dim.dim_value if dim.dim_value != 0 else '?')
                        input_shape = tuple(input_shape)
                except ImportError:
                    logger.warning("未安装onnx库，无法解析ONNX模型详细信息")
                except Exception as e:
                    logger.warning(f"解析ONNX模型 {file_path} 失败: {str(e)}")
            
            # 获取文件时间戳
            timestamp = os.path.getmtime(file_path)
            
            # 创建模型参数
            params = {
                "parameters": model_params,
                "input_shape": input_shape,
                "model_size": round(model_size, 2),
                "file_extension": ext
            }
            
            # 创建并返回ModelInfoData对象（适配新的数据结构）
            model_info = ModelInfoData(
                name=model_name,
                path=file_path,
                model_type=model_type,
                timestamp=timestamp,
                params=params,
                framework=model_type,  # 框架类型与模型类型相同
                task_type="unknown",  # 默认任务类型为未知
                version="1.0"  # 默认版本号
                # 不再需要显式添加namespace参数
            )
            
            # 如果能获取到模型参数，添加一个参数数量的指标
            if model_params is not None:
                params_metric = MetricData(
                    name="Parameters",
                    metric_type="scalar",
                    data={"value": model_params, "unit": ""},
                    source_path=file_path,
                    timestamp=timestamp,
                    description="模型参数总量"
                )
                model_info.add_metric(params_metric)
            
            # 添加模型大小指标
            size_metric = MetricData(
                name="Model Size",
                metric_type="scalar",
                data={"value": round(model_size, 2), "unit": "MB"},
                source_path=file_path,
                timestamp=timestamp,
                description="模型文件大小"
            )
            model_info.add_metric(size_metric)
            
            return model_info
            
        except Exception as e:
            logger.error(f"解析模型文件 {file_path} 失败: {str(e)}")
            raise ValueError(f"解析模型文件失败: {str(e)}")
