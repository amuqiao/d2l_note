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

# 定义一个函数来显示处理器信息
def show_handlers(logger, msg):
    print(f"\n{msg}:")
    print(f"处理器总数: {len(logger.handlers)}")
    console_count = 0
    file_count = 0
    for i, handler in enumerate(logger.handlers):
        handler_type = type(handler).__name__
        print(f"  处理器{i+1}: {handler_type} (ID: {id(handler)})")
        if handler_type == 'StreamHandler':
            console_count += 1
        elif handler_type == 'FileHandler':
            file_count += 1
    print(f"  控制台处理器数量: {console_count}")
    print(f"  文件处理器数量: {file_count}")
    return console_count

# 打印初始状态
d2l_note_logger = logging.getLogger('d2l_note')
print("=== 初始状态 ===")
print(f"root logger处理器数量: {len(root_logger.handlers)}")
print(f"初始d2l_note logger处理器数量: {len(d2l_note_logger.handlers)}")
print(f"初始d2l_note logger propagate设置: {d2l_note_logger.propagate}")

# 导入log_utils模块
print("\n=== 开始导入log_utils模块 ===")
import src.utils.log_utils as log_utils

# 导入后的处理器状态
console_count1 = show_handlers(d2l_note_logger, "导入后的d2l_note logger处理器状态")

# 创建一个简单的函数来检查处理器数量
def check_handlers():
    console_handlers = [h for h in d2l_note_logger.handlers if isinstance(h, logging.StreamHandler)]
    file_handlers = [h for h in d2l_note_logger.handlers if isinstance(h, logging.FileHandler)]
    return len(console_handlers), len(file_handlers)

# 打印处理器数量
after_import_console, after_import_file = check_handlers()
print(f"\n导入后直接检查 - 控制台处理器数量: {after_import_console}")
print(f"导入后直接检查 - 文件处理器数量: {after_import_file}")

# 尝试直接访问日志文件，看看是否会触发额外的处理器创建
print("\n=== 访问日志文件内容 ===")
if os.path.exists("logs/d2l_note.log"):
    with open("logs/d2l_note.log", "r", encoding="utf-8") as f:
        content = f.read()
        print(f"日志文件内容长度: {len(content)} 字符")
        if len(content) > 100:
            print(f"前100个字符: {content[:100]}")
else:
    print("日志文件不存在")

# 再次检查处理器数量
after_file_console, after_file_file = check_handlers()
print(f"\n访问日志文件后 - 控制台处理器数量: {after_file_console}")
print(f"访问日志文件后 - 文件处理器数量: {after_file_file}")

# 创建新的LogUtils实例
print("\n=== 创建新的LogUtils实例 ===")
new_logger = log_utils.LogUtils()

# 检查处理器数量
after_new_console, after_new_file = check_handlers()
print(f"\n创建新实例后 - 控制台处理器数量: {after_new_console}")
print(f"创建新实例后 - 文件处理器数量: {after_new_file}")

# 调用init_logger方法
print("\n=== 调用init_logger方法 ===")
init_result = log_utils.init_logger()

# 再次检查处理器数量
after_init_console, after_init_file = check_handlers()
print(f"\n调用init_logger后 - 控制台处理器数量: {after_init_console}")
print(f"调用init_logger后 - 文件处理器数量: {after_init_file}")

# 最终结论
print("\n=== 最终结论 ===")
final_console, final_file = check_handlers()
print(f"最终控制台处理器数量: {final_console}")
print(f"最终文件处理器数量: {final_file}")
print(f"控制台处理器数量是否为1: {'✓' if final_console == 1 else '✗'}")
print(f"文件处理器数量是否为1: {'✓' if final_file == 1 else '✗'}")

# 打印所有处理器的详细信息
print("\n=== 所有处理器的详细信息 ===")
for i, handler in enumerate(d2l_note_logger.handlers):
    handler_type = type(handler).__name__
    print(f"处理器{i+1}: {handler_type} (ID: {id(handler)})")
    print(f"  级别: {handler.level}")
    print(f"  格式化器: {handler.formatter}")
    if hasattr(handler, 'baseFilename'):
        print(f"  文件路径: {handler.baseFilename}")
    elif hasattr(handler, 'stream'):
        print(f"  流类型: {type(handler.stream).__name__}")
        print(f"  流对象ID: {id(handler.stream)}")

# 验证所有实例共享底层logging对象
print("\n=== 实例共享验证 ===")
print(f"全局logger和new_logger的底层logging对象是否相同: {id(log_utils.logger.logger) == id(new_logger.logger)}")
print(f"全局logger和init_result的底层logging对象是否相同: {id(log_utils.logger.logger) == id(init_result.logger)}")
print(f"所有LogUtils实例共享同一个底层logging对象: {'✓' if id(log_utils.logger.logger) == id(new_logger.logger) == id(init_result.logger) else '✗'}")