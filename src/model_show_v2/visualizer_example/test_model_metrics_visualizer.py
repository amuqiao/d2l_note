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


class TestModelMetricsVisualizer(unittest.TestCase):
    """测试模型指标可视化器"""
    
    def setUp(self):
        """设置测试环境，准备测试数据"""
        # 初始化解析器和可视化器
        self.parser = MetricsFileParser()
        self.visualizer = MetricsVisualizer()
        
        # 测试数据路径
        self.test_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "runs", "run_20250914_040635"
        )
        
        # 确保测试数据文件存在
        self.metrics_file_path = os.path.join(self.test_data_dir, "metrics.json")
        
        # 如果测试数据不存在，创建一个简单的模拟数据文件
        if not os.path.exists(self.metrics_file_path):
            self._create_mock_metrics_file()
        
        # 解析指标文件，获取模型信息
        self.model_info = self.parser.parse(self.metrics_file_path)
        
        # 为比较测试创建一个简单的副本模型信息
        self.comparison_model_info = self._create_comparison_model_info()
    
    def tearDown(self):
        """清理测试环境，删除生成的测试数据文件和所有创建的目录"""
        # 删除生成的模拟指标文件（如果存在）
        if hasattr(self, 'metrics_file_path') and os.path.exists(self.metrics_file_path):
            try:
                os.remove(self.metrics_file_path)
                print(f"已删除生成的测试文件: {self.metrics_file_path}")
            except Exception as e:
                print(f"删除测试文件时出错: {str(e)}")
        
        # 删除测试子目录（如果该目录为空）
        if hasattr(self, 'test_data_dir') and os.path.exists(self.test_data_dir):
            try:
                # 检查目录是否为空
                if not os.listdir(self.test_data_dir):
                    os.rmdir(self.test_data_dir)
                    print(f"已删除空测试子目录: {self.test_data_dir}")
                    
                    # 检查并删除父runs目录（如果为空）
                    runs_dir = os.path.dirname(self.test_data_dir)
                    if os.path.exists(runs_dir) and not os.listdir(runs_dir):
                        os.rmdir(runs_dir)
                        print(f"已删除空runs目录: {runs_dir}")
            except Exception as e:
                print(f"删除测试目录时出错: {str(e)}")
    
    def _create_mock_metrics_file(self):
        """创建模拟的指标数据文件用于测试"""
        # 确保目录存在
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 创建简单的指标数据
        mock_metrics_data = {
            "final_train_loss": 0.2345,
            "final_train_acc": 0.8976,
            "final_test_loss": 0.3456,
            "final_test_acc": 0.8567,
            "best_test_acc": 0.8765,
            "total_training_time": "00:15:23",
            "samples_per_second": 1234.56,
            "training_start_time": "2025-09-14 04:06:35",
            "training_end_time": "2025-09-14 04:21:58",
            "epoch_metrics": [
                {"epoch": 1, "train_loss": 0.7654, "train_acc": 0.6789, "test_acc": 0.6543},
                {"epoch": 2, "train_loss": 0.5432, "train_acc": 0.7654, "test_acc": 0.7123},
                {"epoch": 3, "train_loss": 0.3456, "train_acc": 0.8234, "test_acc": 0.7789},
                {"epoch": 4, "train_loss": 0.2678, "train_acc": 0.8654, "test_acc": 0.8123},
                {"epoch": 5, "train_loss": 0.2345, "train_acc": 0.8976, "test_acc": 0.8567}
            ]
        }
        
        # 写入文件
        import json
        with open(self.metrics_file_path, 'w', encoding='utf-8') as f:
            json.dump(mock_metrics_data, f, indent=2, ensure_ascii=False)
    
    def _create_comparison_model_info(self) -> ModelInfoData:
        """创建用于比较测试的模型信息"""
        # 解析原始指标文件
        comparison_model_info = self.parser.parse(self.metrics_file_path)
        
        # 修改一些指标值，用于比较测试
        comparison_model_info.name = "LeNet-Comparison"
        
        # 修改一些标量指标
        final_test_acc = comparison_model_info.get_metric_by_name("final_test_acc")
        if final_test_acc:
            final_test_acc.data["value"] = 0.8812
        
        best_test_acc = comparison_model_info.get_metric_by_name("best_test_acc")
        if best_test_acc:
            best_test_acc.data["value"] = 0.8945
        
        # 修改训练时间
        comparison_model_info.params["total_training_time"] = "00:12:45"
        
        return comparison_model_info
    
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
        self.assertIn("basic_info", tables, "缺少基本信息表格")
        self.assertIn("scalar_metrics", tables, "缺少标量指标表格")
        self.assertIn("curve_metrics_summary", tables, "缺少曲线指标摘要表格")
        
        # 验证表格内容是否包含关键信息
        text = result["text"]
        self.assertIn(self.model_info.name, text, "可视化文本中未包含模型名称")
        
        # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
        test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
        self.assertTrue(test_acc_related, "可视化文本中未包含测试准确率相关信息")
        
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
        self.assertIn("basic_compare", tables, "缺少基本信息比较表格")
        self.assertIn("scalar_compare", tables, "缺少标量指标比较表格")
        
        # 验证表格内容是否包含关键信息
        text = result["text"]
        self.assertIn(self.model_info.name, text, "比较文本中未包含第一个模型名称")
        self.assertIn(self.comparison_model_info.name, text, "比较文本中未包含第二个模型名称")
        
        # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
        test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
        self.assertTrue(test_acc_related, "比较文本中未包含测试准确率相关信息")
        
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
    unittest.main()