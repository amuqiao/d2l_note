"""
模型文件可视化测试脚本

这个脚本提供了一个直接的方式来测试模型文件的可视化功能，不需要unittest框架。
使用方法：python src/model_show_v2/visualizer_example/test_model_file_visualizer.py
"""
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.model_show_v2.parsers.base_model_parsers import ModelInfoParserRegistry
from src.model_show_v2.visualizers.model_file_visualizer import ModelFileVisualizer


def main():
    """主函数：加载模型文件并进行可视化和比较"""
    print("===== 模型文件可视化测试工具 =====")
    
    # 测试的模型文件路径（与test_model_parser.py保持一致）
    model_file1 = "e:/github_project/d2l_note/runs/run_20250914_040635/best_model_LeNet_acc_0.7860_epoch_9.pth"
    model_file2 = "e:/github_project/d2l_note/runs/run_20250914_040809/best_model_AlexNet_acc_0.9107_epoch_28.pth"
    
    # 验证测试文件是否存在
    for file_path in [model_file1, model_file2]:
        if not os.path.exists(file_path):
            print(f"错误: 测试文件不存在: {file_path}")
            print("请修改脚本中的模型文件路径为实际存在的文件")
            return 1
    
    print(f"\n找到模型文件:")
    print(f"1. {model_file1}")
    print(f"2. {model_file2}")
    
    try:
        # 解析模型文件
        print("\n正在解析模型文件...")
        model_info1 = ModelInfoParserRegistry.parse_file(model_file1, namespace="models")
        model_info2 = ModelInfoParserRegistry.parse_file(model_file2, namespace="models")
        
        if not model_info1 or not model_info2:
            print("错误: 模型文件解析失败")
            return 1
        
        print(f"解析成功! 模型名称: {model_info1.name}, {model_info2.name}")
        print(f"模型类型: {model_info1.model_type}, {model_info2.model_type}")
        
        # 打印模型指标信息
        print(f"\n模型1指标信息:")
        for metric in model_info1.metric_list:
            print(f"  - {metric.name}: {metric.data.get('value', '-')}{metric.data.get('unit', '')} ({metric.description})")
            
        print(f"模型2指标信息:")
        for metric in model_info2.metric_list:
            print(f"  - {metric.name}: {metric.data.get('value', '-')}{metric.data.get('unit', '')} ({metric.description})")
        
        # 创建可视化器实例
        visualizer = ModelFileVisualizer()
        
        # 可视化第一个模型文件
        print("\n===== 可视化第一个模型文件 =====")
        result1 = visualizer.visualize(model_info1, namespace="models")
        if result1 and result1.get("success", False):
            print(result1.get("text"))
        else:
            print(f"可视化失败: {result1.get('message', '未知错误')}")
        
        # 可视化第二个模型文件
        print("\n===== 可视化第二个模型文件 =====")
        result2 = visualizer.visualize(model_info2, namespace="models")
        if result2 and result2.get("success", False):
            print(result2.get("text"))
        else:
            print(f"可视化失败: {result2.get('message', '未知错误')}")
        
        # 比较两个模型文件
        print("\n===== 比较两个模型文件 =====")
        comparison_result = visualizer.compare([model_info1, model_info2], namespace="models")
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