"""
追踪控制台处理器创建过程的脚本
"""
import os
import sys
import logging
import traceback

# 获取项目根目录并添加到sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 清除所有可能的缓存
if 'src.utils.log_utils' in sys.modules:
    del sys.modules['src.utils.log_utils']

# 禁用所有已有的日志处理器
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    handler.close()
    root_logger.removeHandler(handler)

# 创建一个简单的日志处理器来记录追踪信息
trace_handler = logging.StreamHandler(sys.stdout)
trace_handler.setLevel(logging.DEBUG)
trace_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
trace_handler.setFormatter(trace_formatter)
trace_logger = logging.getLogger('trace')
trace_logger.addHandler(trace_handler)
trace_logger.setLevel(logging.DEBUG)

trace_logger.debug("=== 开始追踪LogUtils控制台处理器创建过程 ===")

trace_logger.debug("\n1. 清除d2l_note logger中的处理器")
d2l_logger = logging.getLogger('d2l_note')
for handler in d2l_logger.handlers[:]:
    trace_logger.debug(f"  清理处理器: {type(handler).__name__} (ID: {id(handler)})")
    handler.close()
    d2l_logger.removeHandler(handler)

trace_logger.debug(f"  清理后处理器数量: {len(d2l_logger.handlers)}")

trace_logger.debug("\n2. 准备导入log_utils模块")

try:
    # 创建一个自定义的StreamHandler类来追踪创建过程
    original_stream_handler_init = logging.StreamHandler.__init__
    
    def patched_stream_handler_init(self, *args, **kwargs):
        original_stream_handler_init(self, *args, **kwargs)
        trace_logger.debug(f"  创建了StreamHandler (ID: {id(self)})")
        trace_logger.debug(f"  调用栈:")
        stack = traceback.extract_stack()
        for i, frame in enumerate(stack[:-1]):  # 排除当前帧
            trace_logger.debug(f"    {i+1}. {frame.filename}:{frame.lineno} in {frame.name}")
    
    # 替换原始的__init__方法
    logging.StreamHandler.__init__ = patched_stream_handler_init
    
    trace_logger.debug("\n3. 导入log_utils模块")
    from src.utils import log_utils
    
    # 检查处理器状态
    trace_logger.debug("\n4. 检查导入后的处理器状态")
    d2l_logger = logging.getLogger('d2l_note')
    trace_logger.debug(f"  总处理器数量: {len(d2l_logger.handlers)}")
    
    stream_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.StreamHandler)]
    trace_logger.debug(f"  控制台处理器数量: {len(stream_handlers)}")
    for i, handler in enumerate(stream_handlers):
        trace_logger.debug(f"    控制台处理器{i+1}: ID = {id(handler)}")
    
    file_handlers = [h for h in d2l_logger.handlers if isinstance(h, logging.FileHandler)]
    trace_logger.debug(f"  文件处理器数量: {len(file_handlers)}")
    
finally:
    # 恢复原始的__init__方法
    logging.StreamHandler.__init__ = original_stream_handler_init
    trace_logger.debug("\n=== 追踪结束 ===")