"""
模型指标可视化器测试：验证MetricsVisualizer的功能

功能：测试模型训练指标的可视化和比较功能
"""
import logging
import os
import unittest
from typing import Dict, Any, List

from src.model_show_v2.parsers.model_metrics_parser import MetricsFileParser
from src.model_show_v2.visualizers.model_metrics_visualizer import MetricsVisualizer
from src.model_show_v2.data_models import ModelInfoData
from src.utils.log_utils.log_utils import get_logger

# 设置日志级别为ERROR，减少测试输出
logger = get_logger()
logger.set_global_level(logging.ERROR)

def show_visualization_effect():
    """展示模型指标可视化效果"""
    print("\n=== 模型指标可视化效果展示 ===")
    print("此脚本可以直接运行查看可视化效果，也可以作为单元测试运行")
    print("\npytest和unittest的主要区别：")
    print("1. pytest是第三方测试框架，提供更丰富的功能和更简洁的语法")
    print("2. unittest是Python标准库自带的测试框架")
    print("3. pytest可以运行unittest风格的测试，但反之不行")
    print("4. pytest提供自动发现测试、更丰富的断言、参数化测试等高级功能")
    print("\n" + "="*50 + "\n")
    
    # 初始化解析器和可视化器
    parser = MetricsFileParser()
    visualizer = MetricsVisualizer()
    
    # 测试数据路径 - 使用现有的测试数据文件
    test_data_dir1 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "test_runs", "run_20250914_040635"
    )
    
    test_data_dir2 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "test_runs", "run_20250927_235759"
    )
    
    # 获取指标文件路径
    metrics_file_path1 = os.path.join(test_data_dir1, "metrics.json")
    metrics_file_path2 = os.path.join(test_data_dir2, "metrics.json")
    
    # 检查文件是否存在
    if not os.path.exists(metrics_file_path1):
        print(f"警告：文件不存在: {metrics_file_path1}")
        return
    
    if not os.path.exists(metrics_file_path2):
        print(f"警告：文件不存在: {metrics_file_path2}")
        return
    
    # 解析指标文件，获取模型信息
    model_info1 = parser.parse(metrics_file_path1)
    model_info2 = parser.parse(metrics_file_path2)
    
    # 执行单个模型可视化
    print("\n【单个模型指标可视化 - 模型1】")
    result = visualizer.visualize(model_info1)
    if result["success"]:
        print(result["text"])
    
    # 执行多个模型比较
    print("\n【多模型指标比较】")
    compare_result = visualizer.compare([model_info1, model_info2])
    if compare_result["success"]:
        print(compare_result["text"])
    
    print("\n=== 可视化展示完成 ===")


class TestModelMetricsVisualizer(unittest.TestCase):
    """测试模型指标可视化器"""
    
    def setUp(self):
        """设置测试环境，准备测试数据"""
        # 初始化解析器和可视化器
        self.parser = MetricsFileParser()
        self.visualizer = MetricsVisualizer()
        
        # 测试数据路径 - 使用现有的测试数据文件
        self.test_data_dir1 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "test_runs", "run_20250914_040635"
        )
        
        self.test_data_dir2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "test_runs", "run_20250927_235759"
        )
        
        # 获取指标文件路径
        self.metrics_file_path1 = os.path.join(self.test_data_dir1, "metrics.json")
        self.metrics_file_path2 = os.path.join(self.test_data_dir2, "metrics.json")
        
        # 检查文件是否存在
        if not os.path.exists(self.metrics_file_path1):
            self.fail(f"测试文件不存在: {self.metrics_file_path1}")
        
        if not os.path.exists(self.metrics_file_path2):
            self.fail(f"测试文件不存在: {self.metrics_file_path2}")
        
        # 解析指标文件，获取模型信息
        self.model_info = self.parser.parse(self.metrics_file_path1)
        
        # 为比较测试使用第二个模型数据
        self.comparison_model_info = self.parser.parse(self.metrics_file_path2)
    
    def test_visualize(self):
        """测试可视化功能"""
        # 执行可视化
        result = self.visualizer.visualize(self.model_info)
        
        # 验证可视化结果
        self.assertTrue(result["success"], f"可视化失败: {result.get('message')}")
        self.assertIn("text", result, "可视化结果中缺少text字段")
        self.assertIn("tables", result, "可视化结果中缺少tables字段")
        
        # 验证生成的表格类型
        tables = result["tables"]
        self.assertIn("optimized_metrics", tables, "缺少优化的指标表格")
        
        # 验证表格内容是否包含关键信息
        text = result["text"]
        self.assertIn(self.model_info.name, text, "可视化文本中未包含模型名称")
        
        # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
        test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
        self.assertTrue(test_acc_related, "可视化文本中未包含测试准确率相关信息")
        
        # 验证表格内容是否包含关键信息
        optimized_table = tables["optimized_metrics"]
        # 验证表格中包含模型名称
        self.assertIn(self.model_info.name, str(optimized_table), "表格应包含模型名称")
        # 验证表格中包含关键指标字段
        table_str = str(optimized_table).lower()
        self.assertIn("final_train_loss", table_str, "表格应包含最终训练损失")
        self.assertIn("final_test_acc", table_str, "表格应包含最终测试准确率")
        self.assertIn("best_test_acc", table_str, "表格应包含最佳测试准确率")
        self.assertIn("total_training_time", table_str, "表格应包含总训练时间")
        
        print("\n=== 可视化测试结果 ===")
        print(result["text"])
        print("=== 可视化测试完成 ===\n")
    
    def test_support(self):
        """测试支持判断功能"""
        # 测试支持判断
        is_supported = self.visualizer.support(self.model_info)
        
        # 验证支持判断结果
        self.assertTrue(is_supported, "应该支持该模型指标信息")
        
        # 测试创建一个没有指标数据的模型信息
        import time
        current_time = time.time()
        empty_model_info = ModelInfoData(
            name="EmptyModel",
            path="./empty/path",
            model_type="Unknown",
            timestamp=current_time
        )
        
        # 验证不支持没有指标数据的模型信息
        is_supported_empty = self.visualizer.support(empty_model_info)
        self.assertFalse(is_supported_empty, "不应该支持没有指标数据的模型信息")
    
    def test_compare(self):
        """测试比较功能"""
        # 创建模型信息列表用于比较
        model_infos = [self.model_info, self.comparison_model_info]
        
        # 执行比较
        result = self.visualizer.compare(model_infos)
        
        # 验证比较结果
        self.assertTrue(result["success"], f"比较失败: {result.get('message')}")
        self.assertIn("text", result, "比较结果中缺少text字段")
        self.assertIn("tables", result, "比较结果中缺少tables字段")
        
        # 验证生成的表格类型
        tables = result["tables"]
        self.assertIn("optimized_compare", tables, "缺少优化的比较表格")
        
        # 验证表格内容是否包含关键信息
        text = result["text"]
        self.assertIn(self.model_info.name, text, "比较文本中未包含第一个模型名称")
        self.assertIn(self.comparison_model_info.name, text, "比较文本中未包含第二个模型名称")
        
        # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
        test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
        self.assertTrue(test_acc_related, "比较文本中未包含测试准确率相关信息")
        
        # 验证表格内容是否包含关键信息
        compare_table = tables["optimized_compare"]
        # 验证表格中包含两个模型的名称
        self.assertIn(self.model_info.name, str(compare_table), "表格应包含第一个模型名称")
        self.assertIn(self.comparison_model_info.name, str(compare_table), "表格应包含第二个模型名称")
        # 验证表格中包含关键指标字段
        table_str = str(compare_table).lower()
        self.assertIn("final_train_loss", table_str, "表格应包含最终训练损失")
        self.assertIn("final_test_acc", table_str, "表格应包含最终测试准确率")
        self.assertIn("best_test_acc", table_str, "表格应包含最佳测试准确率")
        self.assertIn("total_training_time", table_str, "表格应包含总训练时间")
        
        print("\n=== 比较测试结果 ===")
        print(result["text"])
        print("=== 比较测试完成 ===\n")
    
    def test_compare_single_model(self):
        """测试比较单个模型时的错误处理"""
        # 创建只有一个模型信息的列表
        model_infos = [self.model_info]
        
        # 执行比较
        result = self.visualizer.compare(model_infos)
        
        # 验证比较结果（应该失败）
        self.assertFalse(result["success"], "比较单个模型时应该失败")
        self.assertIn("需要至少2个模型指标信息", result.get("message", ""), "错误信息不正确")
    
    def test_visualizer_registry_integration(self):
        """测试可视化器与注册中心的集成"""
        from src.model_show_v2.visualizers.base_model_visualizers import ModelVisualizerRegistry
        
        # 检查可视化器是否已注册
        registry = ModelVisualizerRegistry
        visualizer = registry.get_matched_visualizer(self.model_info)
        
        # 验证找到的可视化器是否是MetricsVisualizer的实例
        self.assertIsInstance(visualizer, MetricsVisualizer, "未能从注册中心找到正确的可视化器")
        
        # 直接使用注册中心的visualize_model方法进行测试
        result = registry.visualize_model(self.model_info)
        self.assertTrue(result["success"], "通过注册中心进行可视化失败")


if __name__ == "__main__":
    # 如果直接运行脚本，展示可视化效果
    show_visualization_effect()
    # 如果需要运行pytest测试，可以使用命令行：pytest -v test_model_metrics_visualizer_pytest.py