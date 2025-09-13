import sys
import importlib
import logging
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义一个函数来测试多次导入log_utils模块
def test_multiple_imports():
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    print("=== 测试多次导入log_utils模块 ===")
    
    # 第一次导入
    print("\n--- 第一次导入log_utils模块 ---\n")
    import src.utils.log_utils as log_utils1
    logger1 = log_utils1.logger
    log_utils1.info("第一次导入后的日志记录")
    
    # 检查初始化状态和处理器数量
    initialized1 = log_utils1.is_logger_initialized()
    console_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.StreamHandler)])
    file_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.FileHandler)])
    
    print(f"第一次导入后初始化状态: {initialized1}")
    print(f"第一次导入后的控制台处理器数量: {console_handlers1}")
    print(f"第一次导入后的文件处理器数量: {file_handlers1}")
    print(f"logger1对象ID: {id(logger1)}")
    
    # 清除导入缓存
    del sys.modules['src.utils.log_utils']
    
    # 第二次导入
    print("\n--- 第二次导入log_utils模块 ---\n")
    import src.utils.log_utils as log_utils2
    logger2 = log_utils2.logger
    log_utils2.info("第二次导入后的日志记录")
    
    # 检查初始化状态和处理器数量
    initialized2 = log_utils2.is_logger_initialized()
    console_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.StreamHandler)])
    file_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.FileHandler)])
    
    print(f"第二次导入后初始化状态: {initialized2}")
    print(f"第二次导入后的控制台处理器数量: {console_handlers2}")
    print(f"第二次导入后的文件处理器数量: {file_handlers2}")
    print(f"logger2对象ID: {id(logger2)}")
    
    # 比较两次导入的logger是否为同一个实例
    print(f"\n两次导入的logger是否为同一个实例: {logger1 is logger2}")
    
    # 测试在不同模块中导入
    print("\n--- 测试在不同模块中导入log_utils ---\n")
    # 创建临时模块
    import types
    temp_module = types.ModuleType('temp_module')
    exec("import src.utils.log_utils as log_utils_temp", temp_module.__dict__)
    logger_temp = temp_module.log_utils_temp.logger
    
    print(f"临时模块中的logger对象ID: {id(logger_temp)}")
    print(f"临时模块中的logger是否与第一次导入的logger相同: {logger_temp is logger1}")
    
    # 初始化状态和处理器数量比较
    if logger1 is logger2:
        print("\n结论: 多次导入log_utils模块时，返回的logger是同一个实例。")
        print(f"初始化状态: 第一次={initialized1}, 第二次={initialized2}")
        print(f"控制台处理器数量: 第一次={console_handlers1}, 第二次={console_handlers2}")
        print(f"文件处理器数量: 第一次={file_handlers1}, 第二次={file_handlers2}")
    else:
        print("\n结论: 多次导入log_utils模块时，返回了不同的logger实例。")

if __name__ == "__main__":
    test_multiple_imports()