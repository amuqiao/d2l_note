import sys
import importlib
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义一个函数来测试多次导入log_utils模块
def test_multiple_imports():
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    print("=== 验证LogUtils单例模式修复 ===")
    
    # 第一次导入
    print("\n--- 第一次导入log_utils模块 ---")
    import src.utils.log_utils as log_utils1
    logger1 = log_utils1.logger
    log_utils1.info("第一次导入后的日志记录")
    
    # 显示第一次导入的状态
    print(f"第一次导入后初始化状态: {log_utils1.is_logger_initialized()}")
    print(f"logger1对象ID: {id(logger1)}")
    print(f"LogUtils类ID: {id(log_utils1.LogUtils)}")
    
    # 获取全局单例存储（如果可以访问）
    try:
        global_store = log_utils1._global_singleton_store
        print(f"全局单例存储ID: {id(global_store)}")
        print(f"全局存储中的实例ID: {id(global_store.get('_log_utils_instance'))}")
        print(f"全局存储中的初始化状态: {global_store.get('_log_utils_initialized')}")
    except AttributeError:
        print("无法访问全局单例存储")
    
    # 清除导入缓存
    print("\n--- 清除模块缓存 ---")
    del sys.modules['src.utils.log_utils']
    
    # 第二次导入
    print("\n--- 第二次导入log_utils模块 ---")
    import src.utils.log_utils as log_utils2
    logger2 = log_utils2.logger
    log_utils2.info("第二次导入后的日志记录")
    
    # 显示第二次导入的状态
    print(f"第二次导入后初始化状态: {log_utils2.is_logger_initialized()}")
    print(f"logger2对象ID: {id(logger2)}")
    print(f"LogUtils类ID: {id(log_utils2.LogUtils)}")
    
    # 获取第二次导入的全局单例存储
    try:
        global_store2 = log_utils2._global_singleton_store
        print(f"第二次全局单例存储ID: {id(global_store2)}")
        print(f"第二次全局存储中的实例ID: {id(global_store2.get('_log_utils_instance'))}")
        print(f"第二次全局存储中的初始化状态: {global_store2.get('_log_utils_initialized')}")
    except AttributeError:
        print("无法访问全局单例存储")
    
    # 比较两次导入的logger是否为同一个实例
    print(f"\n两次导入的logger是否为同一个实例: {logger1 is logger2}")
    
    # 检查控制台处理器数量
    import logging
    console_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.StreamHandler)])
    console_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.StreamHandler)])
    file_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.FileHandler)])
    file_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.FileHandler)])
    
    print(f"第一次导入后的控制台处理器数量: {console_handlers1}")
    print(f"第一次导入后的文件处理器数量: {file_handlers1}")
    print(f"第二次导入后的控制台处理器数量: {console_handlers2}")
    print(f"第二次导入后的文件处理器数量: {file_handlers2}")
    
    # 测试使用get_logger方法
    print("\n--- 测试get_logger方法 ---")
    logger3 = log_utils2.LogUtils.get_logger()
    print(f"get_logger获取的logger对象ID: {id(logger3)}")
    print(f"get_logger获取的logger与logger2是否相同: {logger3 is logger2}")
    
    # 手动创建新实例
    print("\n--- 手动创建新实例 ---")
    logger4 = log_utils2.LogUtils()
    print(f"手动创建的logger对象ID: {id(logger4)}")
    print(f"手动创建的logger与logger2是否相同: {logger4 is logger2}")

if __name__ == "__main__":
    test_multiple_imports()