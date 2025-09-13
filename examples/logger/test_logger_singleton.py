from src.utils.logger import Logger

# 测试单例模式
def test_singleton():
    print("===== 测试Logger单例模式 =====")
    
    # 创建多个Logger实例
    logger1 = Logger(auto_init=True)
    logger2 = Logger()
    logger3 = Logger()
    
    # 验证是否返回同一个实例
    print(f"logger1 与 logger2 是同一个实例: {logger1 is logger2}")
    print(f"logger2 与 logger3 是同一个实例: {logger1 is logger3}")
    
    # 输出实例的id，更直观地验证
    print(f"logger1 id: {id(logger1)}")
    print(f"logger2 id: {id(logger2)}")
    print(f"logger3 id: {id(logger3)}")
    
    # 测试get_instance方法
    instance = Logger.get_instance()
    print(f"get_instance() 返回的实例与logger1相同: {instance is logger1}")
    
    # 测试日志功能
    logger1.info("测试日志信息")
    logger2.warning("测试警告信息")
    
    print("===== 测试完成 =====")

if __name__ == "__main__":
    test_singleton()