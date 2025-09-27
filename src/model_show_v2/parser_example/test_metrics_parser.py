"""
指标文件解析器测试用例

功能：测试MetricsFileParser解析metrics.json文件的能力，验证其能否正确提取模型性能指标信息。
使用示例：python -m unittest src.model_show_v2.parser_example.test_metrics_parser
"""
import unittest
import os
from src.model_show_v2.parsers.base_model_parsers import ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData, MetricData


class TestMetricsParser(unittest.TestCase):
    """指标文件解析器测试用例"""
    
    def setUp(self):
        """设置测试环境，准备测试文件路径"""
        # 测试的指标文件路径
        self.metrics_file1 = "e:/github_project/d2l_note/runs/run_20250914_040635/metrics.json"
        
        # 验证测试文件是否存在
        if not os.path.exists(self.metrics_file1):
            self.skipTest(f"测试文件不存在: {self.metrics_file1}")
    
    def test_parse_metrics_file(self):
        """测试指标文件解析功能"""
        # 解析指标文件
        model_info = ModelInfoParserRegistry.parse_file(
            file_path=self.metrics_file1,
            namespace="metrics"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info, ModelInfoData)
        self.assertIsNotNone(model_info.name)
        self.assertEqual(model_info.path, os.path.dirname(self.metrics_file1))
        self.assertIn(model_info.model_type, ["unknown", "CNN-LeNet", "LeNet"])
        self.assertEqual(model_info.framework, "PyTorch")
        self.assertEqual(model_info.task_type, "classification")
        
        # 验证参数是否正确解析
        self.assertIn("total_training_time", model_info.params)
        self.assertIn("samples_per_second", model_info.params)
        self.assertIn("training_start_time", model_info.params)
        self.assertIn("training_end_time", model_info.params)
        
        # 验证指标列表是否正确
        self.assertGreater(len(model_info.metric_list), 0)
        
        # 验证标量指标
        scalar_metrics = model_info.get_metrics_by_type("scalar")
        self.assertGreater(len(scalar_metrics), 0)
        
        # 验证曲线指标
        curve_metrics = model_info.get_metrics_by_type("curve")
        self.assertGreater(len(curve_metrics), 0)
        
        # 验证特定指标是否存在
        self.assertIsNotNone(model_info.get_metric_by_name("Best Test Accuracy"))
        self.assertIsNotNone(model_info.get_metric_by_name("Training Loss"))
        self.assertIsNotNone(model_info.get_metric_by_name("Test Accuracy"))
        
        # 打印解析结果
        print(f"\n指标文件解析结果:")
        print(f"模型名称: {model_info.name}")
        print(f"模型类型: {model_info.model_type}")
        print(f"参数字典: {model_info.params}")
        print(f"框架: {model_info.framework}")
        print(f"任务类型: {model_info.task_type}")
        print(f"指标数量: {len(model_info.metric_list)}")
        print(f"标量指标数量: {len(scalar_metrics)}")
        print(f"曲线指标数量: {len(curve_metrics)}")
        
        # 打印标量指标详情
        print(f"\n标量指标详情:")
        for metric in scalar_metrics:
            print(f"  - {metric.name}: {metric.data.get('value', '-')}{metric.data.get('unit', '')} ({metric.description})")
        
        # 打印曲线指标详情
        print(f"\n曲线指标详情:")
        for metric in curve_metrics:
            print(f"  - {metric.name}: {len(metric.data.get('epochs', []))}个数据点 ({metric.description})")
    
    def test_metric_data_structure(self):
        """测试指标数据结构是否正确"""
        # 解析指标文件
        model_info = ModelInfoParserRegistry.parse_file(
            file_path=self.metrics_file1,
            namespace="metrics"
        )
        
        # 测试标量指标数据结构
        best_test_acc = model_info.get_metric_by_name("Best Test Accuracy")
        if best_test_acc:
            self.assertEqual(best_test_acc.metric_type, "scalar")
            self.assertIn("value", best_test_acc.data)
            self.assertIn("unit", best_test_acc.data)
            self.assertGreater(best_test_acc.data.get("value", 0), 0)
            self.assertEqual(best_test_acc.data.get("unit"), "%")
            
        # 测试曲线指标数据结构
        training_loss = model_info.get_metric_by_name("Training Loss")
        if training_loss:
            self.assertEqual(training_loss.metric_type, "curve")
            self.assertIn("epochs", training_loss.data)
            self.assertIn("values", training_loss.data)
            self.assertIn("unit", training_loss.data)
            self.assertEqual(len(training_loss.data.get("epochs", [])), len(training_loss.data.get("values", [])))
    
    def test_metric_management_methods(self):
        """测试指标管理辅助方法"""
        # 解析指标文件
        model_info = ModelInfoParserRegistry.parse_file(
            file_path=self.metrics_file1,
            namespace="metrics"
        )
        
        # 测试添加指标
        original_count = len(model_info.metric_list)
        
        new_metric = MetricData(
            name="Test Metric",
            metric_type="scalar",
            data={"value": 100, "unit": ""},
            source_path=self.metrics_file1,
            timestamp=os.path.getmtime(self.metrics_file1),
            description="Test metric"
        )
        
        model_info.add_metric(new_metric)
        self.assertEqual(len(model_info.metric_list), original_count + 1)
        
        # 测试指标去重（添加同名指标应替换）
        updated_metric = MetricData(
            name="Test Metric",
            metric_type="scalar",
            data={"value": 200, "unit": ""},
            source_path=self.metrics_file1,
            timestamp=os.path.getmtime(self.metrics_file1),
            description="Updated test metric"
        )
        
        model_info.add_metric(updated_metric)
        self.assertEqual(len(model_info.metric_list), original_count + 1)
        
        # 测试获取指标
        metric = model_info.get_metric_by_name("Test Metric")
        self.assertIsNotNone(metric)
        self.assertEqual(metric.data.get("value"), 200)
        
        # 测试删除指标
        self.assertTrue(model_info.remove_metric("Test Metric"))
        self.assertEqual(len(model_info.metric_list), original_count)
        self.assertIsNone(model_info.get_metric_by_name("Test Metric"))


if __name__ == "__main__":
    unittest.main()