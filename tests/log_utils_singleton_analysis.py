import sys
import importlib
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 定义一个函数来详细分析LogUtils单例模式的行为
def analyze_singleton_behavior():
    print("=== LogUtils单例模式详细分析 ===")
    
    # 清除所有之前的导入
    if 'src.utils.log_utils' in sys.modules:
        del sys.modules['src.utils.log_utils']
    
    print("\n--- 场景1：正常导入并检查单例 ---\n")
    # 第一次导入
    import src.utils.log_utils as log_utils1
    logger1 = log_utils1.logger
    
    # 获取LogUtils类引用
    LogUtilsClass = log_utils1.LogUtils
    
    print(f"log_utils1模块ID: {id(log_utils1)}")
    print(f"logger1对象ID: {id(logger1)}")
    print(f"LogUtils._instance对象ID: {id(LogUtilsClass._instance)}")
    print(f"logger1是否等于LogUtils._instance: {logger1 is LogUtilsClass._instance}")
    
    # 创建一个新的实例
    new_instance = LogUtilsClass()
    print(f"new_instance对象ID: {id(new_instance)}")
    print(f"new_instance是否等于logger1: {new_instance is logger1}")
    print(f"new_instance是否等于LogUtils._instance: {new_instance is LogUtilsClass._instance}")
    
    # 清除导入缓存
    print("\n--- 场景2：清除导入缓存后重新导入 ---\n")
    del sys.modules['src.utils.log_utils']
    
    # 第二次导入
    import src.utils.log_utils as log_utils2
    logger2 = log_utils2.logger
    
    # 获取新导入的LogUtils类引用
    LogUtilsClass2 = log_utils2.LogUtils
    
    print(f"log_utils2模块ID: {id(log_utils2)}")
    print(f"logger2对象ID: {id(logger2)}")
    print(f"LogUtils._instance对象ID: {id(LogUtilsClass2._instance)}")
    print(f"logger2是否等于logger1: {logger2 is logger1}")
    print(f"log_utils1模块是否等于log_utils2模块: {log_utils1 is log_utils2}")
    
    # 检查处理器数量变化
    print("\n--- 场景3：检查多次导入后的处理器数量 ---\n")
    # 检查处理器数量变化
    import logging
    console_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.StreamHandler)])
    file_handlers1 = len([h for h in logger1.logger.handlers if isinstance(h, logging.FileHandler)])
    
    console_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.StreamHandler)])
    file_handlers2 = len([h for h in logger2.logger.handlers if isinstance(h, logging.FileHandler)])
    
    print(f"第一次导入后的控制台处理器数量: {console_handlers1}")
    print(f"第一次导入后的文件处理器数量: {file_handlers1}")
    print(f"第二次导入后的控制台处理器数量: {console_handlers2}")
    print(f"第二次导入后的文件处理器数量: {file_handlers2}")
    
    # 检查全局变量的影响
    print("\n--- 场景4：检查全局变量初始化时机 ---\n")
    # 清除导入缓存
    del sys.modules['src.utils.log_utils']
    
    # 导入但不直接访问logger
    import src.utils.log_utils as log_utils3
    # 获取LogUtils类引用
    LogUtilsClass3 = log_utils3.LogUtils
    
    print(f"导入后但未访问logger时，LogUtils._instance是否为None: {LogUtilsClass3._instance is None}")
    
    # 访问logger
    logger3 = log_utils3.logger
    print(f"访问logger后，LogUtils._instance对象ID: {id(LogUtilsClass3._instance)}")
    print(f"logger3是否等于LogUtils._instance: {logger3 is LogUtilsClass3._instance}")
    
    print("\n=== 分析总结 ===")
    print("1. LogUtils类内部实现了单例模式，通过_instance类变量保证类内部只有一个实例")
    print("2. 但当清除模块导入缓存并重新导入时，会创建新的LogUtils实例")
    print("3. 这是因为模块级别的全局变量'logger = LogUtils()'在每次导入模块时都会执行")
    print("4. 解决方案建议：将全局logger实例的创建延迟到首次使用时，而不是在模块导入时")

if __name__ == "__main__":
    import logging  # 确保在脚本中导入logging
    analyze_singleton_behavior()