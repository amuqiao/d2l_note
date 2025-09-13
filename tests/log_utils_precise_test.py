"""
精确测试LogUtils底层logger和处理器状态的脚本
"""
import os
import sys
import logging

# 获取项目根目录并添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 清除所有可能的缓存
if 'src.utils.log_utils' in sys.modules:
    del sys.modules['src.utils.log_utils']

# 清除root logger中的处理器
root_logger = logging.getLogger()
print(f"初始root logger处理器数量: {len(root_logger.handlers)}")
for handler in root_logger.handlers[:]:
    print(f"  清理root logger处理器: {type(handler).__name__} (ID: {id(handler)})")
    handler.close()
    root_logger.removeHandler(handler)

# 清除d2l_note logger中的处理器
d2l_logger = logging.getLogger('d2l_note')
print(f"初始d2l_note logger处理器数量: {len(d2l_logger.handlers)}")
for handler in d2l_logger.handlers[:]:
    print(f"  清理d2l_note logger处理器: {type(handler).__name__} (ID: {id(handler)})")
    handler.close()
    d2l_logger.removeHandler(handler)

print("\n=== 测试LogUtils实例和处理器状态 ===")

# 导入log_utils模块
print("\n1. 导入log_utils模块:")
from src.utils import log_utils

# 检查logger对象和处理器状态
logger = log_utils.logger
print(f"logger实例ID: {id(logger)}")
print(f"logger底层logging对象: {id(logger.logger)}")

# 详细检查处理器状态
d2l_logger = logging.getLogger('d2l_note')
print(f"d2l_note logger处理器数量: {len(d2l_logger.handlers)}")
# 注意：在Python logging模块中，FileHandler是StreamHandler的子类
# 所以需要特别区分真正的控制台处理器和文件处理器
stream_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
file_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.FileHandler)]
print(f"  控制台处理器数量: {len(stream_handlers)}")
print(f"  文件处理器数量: {len(file_handlers)}")

# 打印每个处理器的详细信息
for i, handler in enumerate(d2l_logger.handlers):
    handler_type = type(handler).__name__
    print(f"  处理器{i+1}: {handler_type} (ID: {id(handler)})")
    
# 创建新的LogUtils实例
print("\n2. 创建新的LogUtils实例:")
new_logger = log_utils.LogUtils()
print(f"new_logger实例ID: {id(new_logger)}")
print(f"new_logger底层logging对象: {id(new_logger.logger)}")

# 再次检查处理器状态
d2l_logger = logging.getLogger('d2l_note')
print(f"创建新实例后d2l_note logger处理器数量: {len(d2l_logger.handlers)}")
stream_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
file_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.FileHandler)]
print(f"  控制台处理器数量: {len(stream_handlers)}")
print(f"  文件处理器数量: {len(file_handlers)}")

# 通过get_logger获取实例
print("\n3. 通过get_logger获取实例:")
get_logger_instance = log_utils.LogUtils.get_logger()
print(f"get_logger_instance实例ID: {id(get_logger_instance)}")
print(f"get_logger_instance底层logging对象: {id(get_logger_instance.logger)}")

# 调用init_logger方法
print("\n4. 调用init_logger方法:")
init_result = log_utils.init_logger(use_timestamp=False)
print(f"init_logger返回实例ID: {id(init_result)}")

# 最后检查处理器状态
d2l_logger = logging.getLogger('d2l_note')
print(f"调用init_logger后d2l_note logger处理器数量: {len(d2l_logger.handlers)}")
stream_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
file_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.FileHandler)]
print(f"  控制台处理器数量: {len(stream_handlers)}")
print(f"  文件处理器数量: {len(file_handlers)}")

# 验证所有实例是否共享同一个底层logging对象
print("\n=== 实例共享验证 ===")
print(f"logger和new_logger的底层logging对象是否相同: {logger.logger is new_logger.logger}")
print(f"logger和get_logger_instance的底层logging对象是否相同: {logger.logger is get_logger_instance.logger}")
print(f"logger和init_result的底层logging对象是否相同: {logger.logger is init_result.logger}")

print("\n=== 结论 ===")
print(f"1. 所有LogUtils实例共享同一个底层logging对象: {'✓' if logger.logger is new_logger.logger and logger.logger is get_logger_instance.logger and logger.logger is init_result.logger else '✗'}")
print(f"2. 控制台处理器数量保持为1个: {'✓' if len(stream_handlers) == 1 else '✗'}")
print(f"3. 文件处理器数量保持为1个: {'✓' if len(file_handlers) == 1 else '✗'}")
print(f"4. 日志系统功能正常: {'✓' if logger.is_initialized() else '✗'}")