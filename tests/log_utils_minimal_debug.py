import sys
import os
import logging

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

# 打印初始状态
d2l_note_logger = logging.getLogger('d2l_note')
print("=== 初始状态 ===")
print(f"root logger处理器数量: {len(root_logger.handlers)}")
print(f"d2l_note logger处理器数量: {len(d2l_note_logger.handlers)}")
print(f"d2l_note logger propagate设置: {d2l_note_logger.propagate}")

# 导入log_utils模块
print("\n=== 导入log_utils模块 ===")
import src.utils.log_utils as log_utils

# 导入后的处理器状态
print("\n=== 导入后的处理器状态 ===")
print(f"处理器总数: {len(d2l_note_logger.handlers)}")

# 直接打印所有处理器的类型和ID
print("\n=== 所有处理器详细信息 ===")
for i, handler in enumerate(d2l_note_logger.handlers):
    print(f"处理器{i+1}: {type(handler).__name__} (ID: {id(handler)})")
    print(f"  StreamHandler类型: {isinstance(handler, logging.StreamHandler)}")
    print(f"  FileHandler类型: {isinstance(handler, logging.FileHandler)}")
    print(f"  Handler基类: {isinstance(handler, logging.Handler)}")
    
    # 打印处理器的所有父类，看是否有继承关系
    bases = [base.__name__ for base in type(handler).__mro__]
    print(f"  所有父类: {bases}")

# 计算处理器类型数量
console_handlers = [h for h in d2l_note_logger.handlers if isinstance(h, logging.StreamHandler)]
file_handlers = [h for h in d2l_note_logger.handlers if isinstance(h, logging.FileHandler)]

print("\n=== 处理器类型统计 ===")
print(f"StreamHandler数量: {len(console_handlers)}")
print(f"FileHandler数量: {len(file_handlers)}")
print(f"处理器总数: {len(d2l_note_logger.handlers)}")

# 直接打印每个处理器的 isinstance 结果
print("\n=== 处理器类型判断 ===")
for i, handler in enumerate(d2l_note_logger.handlers):
    handler_type = type(handler).__name__
    is_stream = isinstance(handler, logging.StreamHandler)
    is_file = isinstance(handler, logging.FileHandler)
    print(f"处理器{i+1} ({handler_type}): StreamHandler={is_stream}, FileHandler={is_file}")

# 检查是否存在自定义的处理器类型
print("\n=== 检查自定义处理器类型 ===")
for i, handler in enumerate(d2l_note_logger.handlers):
    if type(handler).__module__ != 'logging':
        print(f"处理器{i+1}是自定义类型: {type(handler).__module__}.{type(handler).__name__}")
    else:
        print(f"处理器{i+1}是标准类型: {type(handler).__name__}")