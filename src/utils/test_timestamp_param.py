"""测试日志系统的时间戳参数控制功能"""
import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.logger import Logger, init, reset_logger, get_logger


def test_default_initialization():
    """测试默认初始化是否不添加时间戳"""
    print("\n=== 测试1: 默认初始化 ===")
    reset_logger()
    
    # 默认初始化
    logger = get_logger()
    log_file_path = logger.get_log_file_path()
    
    print(f"日志文件路径: {log_file_path}")
    file_name = os.path.basename(log_file_path)
    # 检查文件名是否为固定的'd2l_note.log'
    assert file_name == 'd2l_note.log', f"默认初始化应该使用固定文件名'd2l_note.log'，但得到: {file_name}"
    print("✓ 默认初始化测试通过: 未添加时间戳")
    
    return log_file_path

def test_custom_initialization_with_timestamp():
    """测试自定义初始化（启用时间戳）"""
    print("\n=== 测试2: 自定义初始化（启用时间戳） ===")
    reset_logger()
    
    # 自定义初始化，显式启用时间戳
    logger = Logger.init(use_timestamp=True)
    log_file_path = logger.get_log_file_path()
    
    print(f"日志文件路径: {log_file_path}")
    # 检查文件名是否包含时间戳格式
    file_name = os.path.basename(log_file_path)
    assert 'd2l_note_' in file_name and file_name.endswith('.log'), \
        f"自定义初始化（启用时间戳）应该包含时间戳，但得到: {log_file_path}"
    assert any(char.isdigit() for char in file_name), \
        f"自定义初始化（启用时间戳）应该包含数字时间戳，但得到: {log_file_path}"
    print("✓ 自定义初始化（启用时间戳）测试通过: 已添加时间戳")
    
    return log_file_path

def test_direct_init_function():
    """测试直接调用init函数"""
    print("\n=== 测试3: 直接调用init函数 ===")
    reset_logger()
    
    # 直接调用init函数，显式禁用时间戳
    logger = init(use_timestamp=False)
    log_file_path = logger.get_log_file_path()
    
    print(f"日志文件路径: {log_file_path}")
    # 检查文件名是否为固定的'd2l_note.log'
    assert 'd2l_note.log' in log_file_path, \
        f"init函数（禁用时间戳）应该使用固定文件名，但得到: {log_file_path}"
    print("✓ 直接调用init函数（禁用时间戳）测试通过: 未添加时间戳")
    
    return log_file_path

def test_multiple_instances():
    """测试多个实例的行为"""
    print("\n=== 测试4: 多个实例测试 ===")
    reset_logger()
    
    # 创建第一个实例（默认初始化）
    logger1 = get_logger()
    path1 = logger1.get_log_file_path()
    print(f"第一个实例日志路径: {path1}")
    
    # 重置后创建第二个实例（自定义时间戳）
    reset_logger()
    logger2 = Logger.init(use_timestamp=True)
    path2 = logger2.get_log_file_path()
    print(f"第二个实例日志路径: {path2}")
    
    # 确保两个路径不同
    assert path1 != path2, "不同初始化方式应该产生不同的日志文件路径"
    
    # 检查文件名规则
    assert 'd2l_note.log' in path1, "默认初始化应该使用固定文件名"
    assert 'd2l_note_' in os.path.basename(path2) and any(char.isdigit() for char in os.path.basename(path2)), \
        "自定义初始化（启用时间戳）应该包含时间戳"
    print("✓ 多个实例测试通过")

def main():
    """主测试函数"""
    print("开始测试日志系统时间戳参数控制功能")
    
    try:
        # 运行各项测试
        paths = []
        paths.append(test_default_initialization())
        paths.append(test_custom_initialization_with_timestamp())
        paths.append(test_direct_init_function())
        test_multiple_instances()
        
        # 验证所有测试的日志文件都存在
        for i, path in enumerate(paths, 1):
            assert os.path.exists(path), f"测试{i}的日志文件不存在: {path}"
            print(f"测试{i}的日志文件已确认存在")
        
        print("\n所有测试通过！日志系统时间戳参数控制功能正常工作。")
    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 测试发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理，重置日志系统
        reset_logger()

if __name__ == "__main__":
    main()