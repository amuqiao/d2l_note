"""
简单的日志处理器调试脚本，追踪控制台处理器的创建过程
"""
import os
import sys
import logging

# 获取项目根目录并添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 清除所有可能的缓存
if 'src.utils.log_utils' in sys.modules:
    del sys.modules['src.utils.log_utils']

# 获取root logger并清除所有处理器
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    print(f"清理root logger中的处理器: {type(handler).__name__}")
    handler.close()
    root_logger.removeHandler(handler)

# 清除特定logger的处理器
d2l_logger = logging.getLogger('d2l_note')
for handler in d2l_logger.handlers[:]:
    print(f"清理d2l_note logger中的处理器: {type(handler).__name__}")
    handler.close()
    d2l_logger.removeHandler(handler)

print("\n=== 开始调试LogUtils处理器创建过程 ===")
print(f"初始root logger处理器数量: {len(root_logger.handlers)}")
print(f"初始d2l_note logger处理器数量: {len(d2l_logger.handlers)}")

# 第一次导入log_utils
print("\n1. 导入log_utils模块:")
from src.utils import log_utils

# 检查处理器数量
d2l_logger = logging.getLogger('d2l_note')
print(f"导入后d2l_note logger处理器数量: {len(d2l_logger.handlers)}")

# 查看每个处理器的类型和来源
for i, handler in enumerate(d2l_logger.handlers):
    print(f"  处理器{i+1}: {type(handler).__name__} (ID: {id(handler)})")
    
# 检查logger对象
logger = log_utils.logger
print(f"\nlogger实例ID: {id(logger)}")
print(f"logger底层logging对象: {id(logger.logger)}")

# 尝试手动初始化
print("\n2. 调用init_logger方法:")
logger.init_logger(use_timestamp=False)

# 再次检查处理器数量
d2l_logger = logging.getLogger('d2l_note')
print(f"调用init_logger后处理器数量: {len(d2l_logger.handlers)}")

# 查看每个处理器的类型
for i, handler in enumerate(d2l_logger.handlers):
    print(f"  处理器{i+1}: {type(handler).__name__} (ID: {id(handler)})")
    
print("\n=== 调试总结 ===")
print(f"最终控制台处理器数量: {len([h for h in d2l_logger.handlers if isinstance(h, logging.StreamHandler)])}")
print(f"最终文件处理器数量: {len([h for h in d2l_logger.handlers if isinstance(h, logging.FileHandler)])}")