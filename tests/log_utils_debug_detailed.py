import sys
import importlib
import logging
import os
import types

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义一个分析函数来检查logger实例和处理器
def analyze_logger(logger, name="logger"):
    print(f"\n=== 分析 {name} ===")
    print(f"LogUtils实例ID: {id(logger)}")
    print(f"底层logging对象ID: {id(logger.logger)}")
    print(f"是否初始化: {logger.is_initialized()}")
    print(f"处理器总数: {len(logger.logger.handlers)}")
    
    # 分析各种类型的处理器
    console_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.StreamHandler)]
    file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
    
    print(f"控制台处理器数量: {len(console_handlers)}")
    for i, handler in enumerate(console_handlers):
        print(f"  控制台处理器{i+1} ID: {id(handler)}")
    
    print(f"文件处理器数量: {len(file_handlers)}")
    for i, handler in enumerate(file_handlers):
        print(f"  文件处理器{i+1} ID: {id(handler)}")
        if hasattr(handler, 'baseFilename'):
            print(f"    日志文件: {handler.baseFilename}")

# 场景1: 正常导入和清除模块缓存
def test_module_reimport():
    print("\n===== 场景1: 测试模块重新导入 ======")
    
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    # 检查LogUtils类是否存在
    print("初始状态下，LogUtils类是否在全局命名空间: 'LogUtils' in globals() =", 'LogUtils' in globals())
    
    # 第一次导入
    print("\n--- 第一次导入log_utils模块 ---")
    import src.utils.log_utils as log_utils1
    logger1 = log_utils1.logger
    
    # 检查导入后的状态
    print("第一次导入后，LogUtils类是否在模块中: 'LogUtils' in log_utils1.__dict__ =", 'LogUtils' in log_utils1.__dict__)
    print(f"第一次导入的模块对象ID: {id(log_utils1)}")
    
    # 分析第一次导入的logger
    analyze_logger(logger1, "第一次导入的logger")
    
    # 清除模块缓存
    print("\n--- 清除模块缓存 ---")
    module_id_before = id(log_utils1)
    log_utils_class_id_before = id(log_utils1.LogUtils)
    del sys.modules['src.utils.log_utils']
    print(f"清除前模块ID: {module_id_before}")
    print(f"清除前LogUtils类ID: {log_utils_class_id_before}")
    
    # 第二次导入
    print("\n--- 第二次导入log_utils模块 ---")
    import src.utils.log_utils as log_utils2
    logger2 = log_utils2.logger
    
    # 检查第二次导入后的状态
    print(f"第二次导入的模块对象ID: {id(log_utils2)}")
    print(f"第二次导入的LogUtils类ID: {id(log_utils2.LogUtils)}")
    print(f"两次导入的模块是否相同: {log_utils1 is log_utils2}")
    print(f"两次导入的LogUtils类是否相同: {log_utils1.LogUtils is log_utils2.LogUtils}")
    print(f"两次导入的logger是否相同: {logger1 is logger2}")
    
    # 分析第二次导入的logger
    analyze_logger(logger2, "第二次导入的logger")

# 场景2: 测试不同方式获取实例
def test_different_instance_creation():
    print("\n===== 场景2: 测试不同方式获取实例 ======")
    
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    # 导入模块
    import src.utils.log_utils as log_utils
    
    # 方式1: 使用全局logger实例
    global_logger = log_utils.logger
    print("\n--- 方式1: 使用全局logger实例 ---")
    analyze_logger(global_logger, "全局logger")
    
    # 方式2: 直接创建LogUtils实例
    direct_logger = log_utils.LogUtils()
    print("\n--- 方式2: 直接创建LogUtils实例 ---")
    analyze_logger(direct_logger, "直接创建的logger")
    
    # 方式3: 使用get_logger类方法
    get_logger = log_utils.LogUtils.get_logger()
    print("\n--- 方式3: 使用get_logger类方法 ---")
    analyze_logger(get_logger, "get_logger获取的logger")
    
    # 比较各种方式获取的实例
    print("\n--- 实例比较 ---")
    print(f"全局logger与直接创建的logger是否相同: {global_logger is direct_logger}")
    print(f"全局logger与get_logger获取的logger是否相同: {global_logger is get_logger}")
    print(f"直接创建的logger与get_logger获取的logger是否相同: {direct_logger is get_logger}")

# 场景3: 测试全局变量初始化时机
def test_global_init_timing():
    print("\n===== 场景3: 测试全局变量初始化时机 ======")
    
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    # 创建一个临时模块来捕获导入过程
    print("创建临时模块来监控导入过程...")
    temp_module = types.ModuleType('import_monitor')
    temp_module.__dict__['imports'] = []
    
    # 定义一个导入监控函数
    def monitor_import(name, *args, **kwargs):
        temp_module.imports.append(name)
        return original_import(name, *args, **kwargs)
    
    # 保存原始的__import__函数并替换它
    original_import = __import__
    builtins.__import__ = monitor_import
    
    try:
        # 导入log_utils模块
        print("\n开始导入log_utils模块...")
        import src.utils.log_utils as log_utils
        print("log_utils模块导入完成")
        
        # 检查全局logger是否已创建
        print(f"\n导入后，全局logger是否已初始化: {log_utils.logger.is_initialized()}")
        
    finally:
        # 恢复原始的__import__函数
        builtins.__import__ = original_import

# 场景4: 分析处理器创建和清理
def test_handler_management():
    print("\n===== 场景4: 分析处理器创建和清理 ======")
    
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    # 导入模块
    import src.utils.log_utils as log_utils
    logger = log_utils.logger
    
    # 记录初始处理器状态
    print("\n--- 初始状态 ---")
    analyze_logger(logger, "初始logger")
    
    # 手动添加一个额外的控制台处理器
    print("\n--- 手动添加额外的控制台处理器 ---")
    extra_console = logging.StreamHandler(sys.stdout)
    extra_console.setLevel(logging.DEBUG)
    logger.logger.addHandler(extra_console)
    analyze_logger(logger, "添加额外处理器后的logger")
    
    # 重新初始化日志系统
    print("\n--- 重新初始化日志系统 ---")
    reinit_logger = log_utils.init_logger()
    analyze_logger(reinit_logger, "重新初始化后的logger")
    
    # 检查处理器是否被清理
    print("\n--- 检查处理器清理 ---")
    console_handlers_after = [h for h in reinit_logger.logger.handlers if isinstance(h, logging.StreamHandler)]
    print(f"重新初始化后的控制台处理器数量: {len(console_handlers_after)}")
    print(f"额外添加的处理器是否还存在: {extra_console in console_handlers_after}")

if __name__ == "__main__":
    # 导入builtins模块
    import builtins
    
    print("===== LogUtils单例模式和处理器管理详细调试 ======")
    
    # 运行所有测试场景
    test_module_reimport()
    test_different_instance_creation()
    test_global_init_timing()
    test_handler_management()
    
    print("\n===== 调试完成 ======")