"""模型信息解析器包"""

# 导入所有解析器模块，触发装饰器注册
from . import config_parser
from . import metrics_parser
from . import model_parser

# 公开需要的类和函数
from .base_parsers import BaseModelInfoParser, ModelInfoParserRegistry
