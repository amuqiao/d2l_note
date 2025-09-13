import sys
import os
import logging
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 清除所有可能的缓存
if 'src.utils.log_utils' in sys.modules:
    del sys.modules['src.utils.log_utils']

# 禁用所有已有的日志处理器
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# 配置root logger，确保没有处理器
root_logger.setLevel(logging.CRITICAL + 1)

# 创建一个日志记录器来记录我们的调试信息
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('DEBUG: %(message)s')
console_handler.setFormatter(console_formatter)
debug_logger.addHandler(console_handler)
debug_logger.propagate = False  # 不传播到root logger

# 打印初始状态
debug_logger.debug("=== 初始状态 ===")
debug_logger.debug(f"root logger处理器数量: {len(root_logger.handlers)}")

d2l_note_logger = logging.getLogger('d2l_note')
debug_logger.debug(f"初始d2l_note logger处理器数量: {len(d2l_note_logger.handlers)}")
debug_logger.debug(f"初始d2l_note logger propagate设置: {d2l_note_logger.propagate}")

# 定义一个函数来显示处理器信息
def show_handlers(logger, msg):
    debug_logger.debug(f"\n{msg}:")
    debug_logger.debug(f"处理器总数: {len(logger.handlers)}")
    console_count = 0
    file_count = 0
    for i, handler in enumerate(logger.handlers):
        handler_type = type(handler).__name__
        debug_logger.debug(f"  处理器{i+1}: {handler_type} (ID: {id(handler)})")
        if handler_type == 'StreamHandler':
            console_count += 1
        elif handler_type == 'FileHandler':
            file_count += 1
    debug_logger.debug(f"  控制台处理器数量: {console_count}")
    debug_logger.debug(f"  文件处理器数量: {file_count}")

# 开始导入log_utils模块
debug_logger.debug("\n=== 开始导入log_utils模块 ===")

# 导入log_utils模块
try:
    # 导入模块前的处理器状态
    show_handlers(d2l_note_logger, "导入前的d2l_note logger处理器状态")
    
    # 导入log_utils模块
    debug_logger.debug("开始导入src.utils.log_utils")
    import src.utils.log_utils as log_utils
    debug_logger.debug("导入完成")
    
    # 导入后的处理器状态
    show_handlers(d2l_note_logger, "导入后的d2l_note logger处理器状态")
    
    # 检查全局logger对象
    if hasattr(log_utils, 'logger'):
        debug_logger.debug(f"\n全局logger实例ID: {id(log_utils.logger)}")
        debug_logger.debug(f"全局logger的底层logging对象ID: {id(log_utils.logger.logger)}")
        debug_logger.debug(f"全局logger的logger.name: {log_utils.logger.logger.name}")
    
    # 检查_logger_initialized状态
    if hasattr(log_utils, '_global_logger_initialized'):
        debug_logger.debug(f"全局logger初始化状态: {log_utils._global_logger_initialized}")
    
    # 创建新的LogUtils实例
    debug_logger.debug("\n=== 创建新的LogUtils实例 ===")
    new_logger = log_utils.LogUtils()
    show_handlers(d2l_note_logger, "创建新实例后的d2l_note logger处理器状态")
    debug_logger.debug(f"新实例ID: {id(new_logger)}")
    debug_logger.debug(f"新实例的底层logging对象ID: {id(new_logger.logger)}")
    
    # 调用init_logger方法
    debug_logger.debug("\n=== 调用init_logger方法 ===")
    init_result = log_utils.init_logger()
    show_handlers(d2l_note_logger, "调用init_logger后的d2l_note logger处理器状态")
    debug_logger.debug(f"init_logger返回实例ID: {id(init_result)}")
    debug_logger.debug(f"init_logger返回实例的底层logging对象ID: {id(init_result.logger)}")
    
    # 验证所有实例共享底层logging对象
    debug_logger.debug("\n=== 实例共享验证 ===")
    if hasattr(log_utils, 'logger'):
        debug_logger.debug(f"全局logger和new_logger的底层logging对象是否相同: {id(log_utils.logger.logger) == id(new_logger.logger)}")
        debug_logger.debug(f"全局logger和init_result的底层logging对象是否相同: {id(log_utils.logger.logger) == id(init_result.logger)}")
    
    # 最终结论
    debug_logger.debug("\n=== 最终结论 ===")
    console_count = len([h for h in d2l_note_logger.handlers if isinstance(h, logging.StreamHandler)])
    file_count = len([h for h in d2l_note_logger.handlers if isinstance(h, logging.FileHandler)])
    debug_logger.debug(f"最终控制台处理器数量: {console_count}")
    debug_logger.debug(f"最终文件处理器数量: {file_count}")
    debug_logger.debug(f"控制台处理器数量是否为1: {'是' if console_count == 1 else '否'}")
    
    # 打印所有处理器的详细信息
    debug_logger.debug("\n=== 所有处理器的详细信息 ===")
    for i, handler in enumerate(d2l_note_logger.handlers):
        handler_type = type(handler).__name__
        debug_logger.debug(f"处理器{i+1}: {handler_type} (ID: {id(handler)})")
        debug_logger.debug(f"  级别: {handler.level}")
        debug_logger.debug(f"  格式化器: {handler.formatter}")
        if hasattr(handler, 'baseFilename'):
            debug_logger.debug(f"  文件路径: {handler.baseFilename}")
        elif hasattr(handler, 'stream'):
            debug_logger.debug(f"  流类型: {type(handler.stream).__name__}")
            
except Exception as e:
    debug_logger.debug(f"导入过程中发生错误: {str(e)}")
    debug_logger.debug(f"错误堆栈: {traceback.format_exc()}")

# 清理资源
for handler in debug_logger.handlers[:]:
    handler.close()
    debug_logger.removeHandler(handler)