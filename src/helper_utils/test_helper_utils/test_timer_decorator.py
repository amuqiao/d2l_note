#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用tool_registry注册和使用timer_decorator中的time_it装饰器的简化示例
"""

# 导入工具注册中心（使用相对导入）
from tool_registry import ToolRegistry
# 导入timer_decorator中的time_it装饰器
from timer_decorator import time_it


def main():
    """主函数，演示注册、获取和使用time_it装饰器"""
    print("\n========================================")
    print("          工具注册中心使用示例")
    print("========================================")
    
    # 1. 注册time_it装饰器到工具注册中心
    print("1. 注册time_it装饰器到工具注册中心...")
    ToolRegistry.register("time_it", time_it)
    print("✅ time_it装饰器已成功注册到工具注册中心")
    
    # 2. 显示所有已注册的工具
    print("\n2. 当前已注册的工具列表:")
    tools = ToolRegistry.list_tools()
    for tool in tools:
        print(f"  - {tool}")
    
    # 3. 从工具注册中心获取time_it装饰器
    print("\n3. 从工具注册中心获取time_it装饰器...")
    timer_decorator = ToolRegistry.get("time_it")
    print("✅ 成功获取time_it装饰器")
    
    # 4. 使用获取的装饰器装饰测试函数
    print("\n4. 使用time_it装饰器装饰测试函数...")
    
    @timer_decorator
    def test_function(seconds):
        """测试函数，模拟耗时操作"""
        import time
        time.sleep(seconds)
        return f"完成 {seconds} 秒的等待"
    
    # 5. 调用被装饰的函数进行测试
    print("\n5. 调用被装饰的测试函数...")
    result = test_function(1)
    print(f"函数返回结果: {result}")
    
    print("\n========================================")
    print("           示例运行完成")
    print("========================================")


if __name__ == "__main__":
    main()