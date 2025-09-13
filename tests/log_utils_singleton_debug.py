import sys
import os
import importlib
import logging

# 确保我们从正确的路径导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def analyze_logger(logger, name="Default"):
    """分析日志实例的属性和处理器"""
    print(f"\n=== {name} Logger Analysis ===")
    print(f"Logger instance ID: {id(logger)}")
    print(f"Logger._initialized: {logger._initialized}")
    print(f"Logger.logger (logging object) ID: {id(logger.logger)}")
    print(f"Number of handlers: {len(logger.logger.handlers)}")
    
    # 分析每个处理器
    for i, handler in enumerate(logger.logger.handlers):
        handler_type = type(handler).__name__
        print(f"  Handler {i+1}: {handler_type}")
        if isinstance(handler, logging.FileHandler):
            print(f"    - File path: {handler.baseFilename}")
        
        # 检查处理器级别
        print(f"    - Level: {logging.getLevelName(handler.level)}")


def test_scenario_1():
    """场景1: 正常导入模块并获取logger"""
    print("\n\n=== 场景1: 正常导入模块并获取logger ===")
    
    # 导入模块
    import src.utils.log_utils as log_utils1
    
    # 获取默认logger
    logger1 = log_utils1.logger
    analyze_logger(logger1, "First import")
    
    # 通过get_logger方法获取logger
    logger2 = log_utils1.LogUtils.get_logger()
    analyze_logger(logger2, "get_logger()")
    
    # 直接创建新实例
    logger3 = log_utils1.LogUtils()
    analyze_logger(logger3, "New instance")
    
    print(f"\nAre all instances the same object? {logger1 is logger2 is logger3}")


def test_scenario_2():
    """场景2: 清除模块缓存后重新导入"""
    print("\n\n=== 场景2: 清除模块缓存后重新导入 ===")
    
    # 首次导入
    import src.utils.log_utils as log_utils1
    logger1 = log_utils1.logger
    analyze_logger(logger1, "Before reload")
    
    # 清除模块缓存
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
        print("Module cache cleared.")
    
    # 重新导入
    import src.utils.log_utils as log_utils2
    logger2 = log_utils2.logger
    analyze_logger(logger2, "After reload")
    
    print(f"\nAre instances from different imports the same? {logger1 is logger2}")
    print(f"Are LogUtils classes the same? {log_utils1.LogUtils is log_utils2.LogUtils}")


def test_scenario_3():
    """场景3: 多次调用init_logger函数"""
    print("\n\n=== 场景3: 多次调用init_logger函数 ===")
    
    # 导入模块
    import src.utils.log_utils as log_utils
    
    # 获取初始logger
    logger1 = log_utils.logger
    analyze_logger(logger1, "Initial")
    
    # 第一次调用init_logger
    logger2 = log_utils.init_logger()
    analyze_logger(logger2, "After init_logger() #1")
    
    # 第二次调用init_logger
    logger3 = log_utils.init_logger()
    analyze_logger(logger3, "After init_logger() #2")
    
    print(f"\nAre all instances the same object? {logger1 is logger2 is logger3}")


def test_scenario_4():
    """场景4: 测试处理器清除机制"""
    print("\n\n=== 场景4: 测试处理器清除机制 ===")
    
    # 导入模块
    import src.utils.log_utils as log_utils
    
    # 获取默认logger
    logger = log_utils.logger
    analyze_logger(logger, "Initial")
    
    # 手动添加一个额外的处理器
    extra_handler = logging.StreamHandler()
    extra_handler.setLevel(logging.DEBUG)
    logger.logger.addHandler(extra_handler)
    print(f"Added an extra handler. Total handlers: {len(logger.logger.handlers)}")
    
    # 重新初始化实例
    new_logger = log_utils.LogUtils()
    analyze_logger(new_logger, "After re-initialization")
    
    print(f"\nAre instances the same? {logger is new_logger}")


def main():
    print("Starting LogUtils singleton pattern debug test...")
    
    # 运行所有测试场景
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()
    
    print("\n\n=== 测试总结 ===")
    print("1. LogUtils类内部通过__new__方法实现了单例模式")
    print("2. 在同一模块实例内，多次获取LogUtils实例都会返回相同对象")
    print("3. 清除模块缓存后重新导入会创建新的LogUtils类定义和实例")
    print("4. 模块级全局变量logger = LogUtils()在每次导入时都会执行")
    print("5. __init__方法中的handlers.clear()机制可以防止处理器重复添加")


if __name__ == "__main__":
    main()