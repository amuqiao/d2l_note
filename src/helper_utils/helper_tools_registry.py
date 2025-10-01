import sys
from typing import Dict, Callable, Any, List
from src.helper_utils.matplotlib_tools import setup_matplotlib_font
from src.helper_utils.timer_decorator import time_it


class ToolRegistry:
    """工具注册中心：管理所有辅助工具，支持注册、调用和列表查看"""
    _tools: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, tool_func: Callable) -> None:
        """注册一个辅助工具"""
        cls._tools[name] = tool_func

    @classmethod
    def get(cls, name: str) -> Callable:
        """获取已注册的工具"""
        if name not in cls._tools:
            raise KeyError(f"工具 '{name}' 未注册")
        return cls._tools[name]

    @classmethod
    def call(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """调用指定的工具，支持传入参数"""
        tool = cls.get(name)
        return tool(*args, **kwargs)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """列出所有已注册的工具名称"""
        return list(cls._tools.keys())
    
    @classmethod
    def get_tool_info(cls) -> Dict[str, str]:
        """获取所有工具的详细信息（名称和函数名）"""
        return {name: func.__name__ for name, func in cls._tools.items()}


# 注册测试工具
def test_tool1():
    """无参数的测试工具"""
    print("测试工具1被调用成功（无参数）")

def test_tool2(message: str, count: int = 1):
    """带参数的测试工具
    Args:
        message: 要打印的消息
        count: 打印次数，默认1次
    """
    for _ in range(count):
        print(f"测试工具2被调用成功，消息：{message}")


# 初始化注册
ToolRegistry.register("test_tool1", test_tool1)
ToolRegistry.register("test_tool2", test_tool2)
ToolRegistry.register("setup_font", setup_matplotlib_font)
ToolRegistry.register("time_it", time_it)


if __name__ == "__main__":
    """测试ToolRegistry的功能"""
    print("开始测试工具注册中心...\n")
    
    # 测试1：列出所有注册工具
    print("测试1：列出所有注册工具")
    all_tools = ToolRegistry.list_tools()
    print(f"已注册的工具列表: {all_tools}")
    print()
    
    # 测试2：获取工具详细信息
    print("测试2：获取工具详细信息")
    tool_info = ToolRegistry.get_tool_info()
    for name, func_name in tool_info.items():
        print(f"工具名称: {name}, 对应函数: {func_name}")
    print()
    
    # 测试3：调用已注册的测试工具
    print("测试3：调用测试工具")
    try:
        # 调用无参数工具
        ToolRegistry.call("test_tool1")
        
        # 调用带参数工具（传递位置参数）
        ToolRegistry.call("test_tool2", "这是位置参数消息")
        
        # 调用带参数工具（传递关键字参数）
        ToolRegistry.call("test_tool2", message="这是关键字参数消息", count=2)
    except KeyError as e:
        print(f"调用工具失败: {e}")
    
    print("\n所有测试完成")
    