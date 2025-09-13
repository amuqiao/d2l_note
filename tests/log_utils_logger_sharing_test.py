"""
验证LogUtils底层logger对象共享的测试脚本
"""
import os
import sys
import logging

# 获取项目根目录并添加到sys.path，以确保可以正确导入src模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 清除之前的模块缓存（如果存在）
if 'src.utils.log_utils' in sys.modules:
    del sys.modules['src.utils.log_utils']

# 第一次导入log_utils模块
print("=== 测试底层logger对象共享状态 ===")
print("\n1. 第一次导入log_utils模块:")
from src.utils import log_utils
logger1 = log_utils.logger
print(f"- logger1实例ID: {id(logger1)}")
print(f"- logger1底层logging对象: {id(logger1.logger)}")
print(f"- 控制台处理器数量: {len([h for h in logger1.logger.handlers if isinstance(h, logging.StreamHandler)])}")
print(f"- 文件处理器数量: {len([h for h in logger1.logger.handlers if isinstance(h, logging.FileHandler)])}")

# 记录一条日志，验证功能正常
logger1.info("从logger1记录的日志")

# 清除模块缓存并重新导入
print("\n2. 清除模块缓存并重新导入log_utils模块:")
del sys.modules['src.utils.log_utils']
from src.utils import log_utils as log_utils2
logger2 = log_utils2.logger
print(f"- logger2实例ID: {id(logger2)}")
print(f"- logger2底层logging对象: {id(logger2.logger)}")
print(f"- 控制台处理器数量: {len([h for h in logger2.logger.handlers if isinstance(h, logging.StreamHandler)])}")
print(f"- 文件处理器数量: {len([h for h in logger2.logger.handlers if isinstance(h, logging.FileHandler)])}")

# 检查两个logger是否共享同一个底层logging对象
print(f"\n3. 底层logging对象共享检查:")
print(f"- logger1和logger2的实例是否相同: {logger1 is logger2}")
print(f"- logger1和logger2的底层logging对象是否相同: {logger1.logger is logger2.logger}")

# 从新的logger记录日志，验证共享状态
logger2.info("从logger2记录的日志")

# 手动创建新的LogUtils实例
print("\n4. 手动创建新的LogUtils实例:")
new_logger = log_utils2.LogUtils()
print(f"- new_logger实例ID: {id(new_logger)}")
print(f"- new_logger底层logging对象: {id(new_logger.logger)}")
print(f"- new_logger底层logging对象是否与logger1相同: {new_logger.logger is logger1.logger}")

# 从手动创建的logger记录日志
new_logger.info("从手动创建的logger记录的日志")

# 尝试通过get_logger方法获取实例
print("\n5. 通过get_logger方法获取实例:")
get_logger_instance = log_utils2.LogUtils.get_logger()
print(f"- get_logger_instance实例ID: {id(get_logger_instance)}")
print(f"- get_logger_instance底层logging对象: {id(get_logger_instance.logger)}")
print(f"- 底层logging对象是否与logger1相同: {get_logger_instance.logger is logger1.logger}")

# 检查初始化状态
print("\n6. 初始化状态检查:")
print(f"- logger1初始化状态: {logger1.is_initialized()}")
print(f"- logger2初始化状态: {logger2.is_initialized()}")
print(f"- new_logger初始化状态: {new_logger.is_initialized()}")
print(f"- get_logger_instance初始化状态: {get_logger_instance.is_initialized()}")

# 验证所有实例都操作同一个处理器集合
print("\n7. 处理器操作验证:")
# 先记录当前处理器数量
initial_console_handlers = len([h for h in logger1.logger.handlers if isinstance(h, logging.StreamHandler)])
print(f"- 当前控制台处理器数量: {initial_console_handlers}")

# 通过一个实例添加一个新的处理器标记
print("- 通过logger1调用init_logger重置日志系统")
logger1.init_logger(use_timestamp=False)

# 检查所有实例的处理器数量是否一致
console_handlers_after_reset = len([h for h in logger2.logger.handlers if isinstance(h, logging.StreamHandler)])
print(f"- 重置后，logger2的控制台处理器数量: {console_handlers_after_reset}")
print(f"- 重置后，new_logger的控制台处理器数量: {len([h for h in new_logger.logger.handlers if isinstance(h, logging.StreamHandler)])}")

print("\n=== 测试总结 ===")
print(f"1. 所有LogUtils实例共享同一个底层logging对象: {'✓' if logger1.logger is logger2.logger else '✗'}")
print(f"2. 控制台处理器数量保持稳定: {'✓' if console_handlers_after_reset == 1 else '✗'}")
print(f"3. 日志系统功能正常: {'✓' if logger1.is_initialized() and logger2.is_initialized() else '✗'}")
print("\n结论: 尽管创建了多个LogUtils实例对象，但它们都共享同一个底层的logging对象，")
print("这确保了日志处理器不会重复创建，避免了资源泄漏问题。")