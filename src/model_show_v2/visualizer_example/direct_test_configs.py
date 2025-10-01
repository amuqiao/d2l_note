"""
直接测试脚本：加载并可视化指定的运行目录配置文件

这个脚本提供了一个更直接的方式来测试配置文件的可视化功能，不需要unittest框架。
使用方法：python src/model_show/visualizer_example/direct_test_configs.py
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model_show_v2.parsers.base_model_parsers import ModelInfoParserRegistry
from src.model_show_v2.visualizers.config_file_visualizer import ConfigFileVisualizer


def main():
    """主函数：加载配置文件并进行可视化和比较"""
    print("===== 配置文件可视化测试工具 =====")
    
    # 测试的运行目录路径
    run_dir1 = "e:/github_project/d2l_note/runs/run_20250914_040635"
    run_dir2 = "e:/github_project/d2l_note/runs/run_20250914_040809"
    
    # 构建配置文件路径
    config_file1 = os.path.join(run_dir1, "config.json")
    config_file2 = os.path.join(run_dir2, "config.json")
    
    # 验证测试文件是否存在
    for file_path in [config_file1, config_file2]:
        if not os.path.exists(file_path):
            print(f"错误: 测试文件不存在: {file_path}")
            return 1
    
    print(f"\n找到配置文件:")
    print(f"1. {config_file1}")
    print(f"2. {config_file2}")
    
    try:
        # 解析配置文件
        print("\n正在解析配置文件...")
        model_info1 = ModelInfoParserRegistry.parse_file(config_file1, namespace="config")
        model_info2 = ModelInfoParserRegistry.parse_file(config_file2, namespace="config")
        
        if not model_info1 or not model_info2:
            print("错误: 配置文件解析失败")
            return 1
        
        print(f"解析成功! 模型名称: {model_info1.name}, {model_info2.name}")
        
        # 创建可视化器实例
        visualizer = ConfigFileVisualizer()
        
        # 可视化第一个配置文件
        print("\n===== 可视化第一个配置文件 =====")
        result1 = visualizer.visualize(model_info1, namespace="config")
        if result1 and result1.get("success", False):
            print(result1.get("text"))
        else:
            print(f"可视化失败: {result1.get('message', '未知错误')}")
        
        # 可视化第二个配置文件
        print("\n===== 可视化第二个配置文件 =====")
        result2 = visualizer.visualize(model_info2, namespace="config")
        if result2 and result2.get("success", False):
            print(result2.get("text"))
        else:
            print(f"可视化失败: {result2.get('message', '未知错误')}")
        
        # 比较两个配置文件
        print("\n===== 比较两个配置文件 =====")
        comparison_result = visualizer.compare([model_info1, model_info2], namespace="config")
        if comparison_result and comparison_result.get("success", False):
            print(comparison_result.get("text"))
        else:
            print(f"比较失败: {comparison_result.get('message', '未知错误')}")
        
        print("\n===== 测试完成 =====")
        return 0
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())