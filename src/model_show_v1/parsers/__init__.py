"""模型信息解析器包"""

# 导入所有解析器模块，触发装饰器注册
from . import model_config_parser
from . import model_metrics_parser
from . import model_file_parser

# 公开需要的类和函数
from .base_model_parsers import BaseModelInfoParser, ModelInfoParserRegistry
