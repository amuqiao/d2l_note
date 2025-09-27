"""
指标文件解析器测试用例

功能：测试MetricsFileParser解析metrics.json文件的能力，验证其能否正确提取模型准确率、损失值等关键指标数据。
使用示例：python -m unittest src.model_show.parser_example.test_metrics_parser
"""
import os
import unittest
from datetime import datetime

# 导入解析器和数据模型
from src.model_show.parser_registry import parse_model_info
from src.model_show.data_models import ModelInfoData


class TestMetricsParser(unittest.TestCase):
    """指标文件解析器测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 测试用的指标文件路径
        self.test_metrics_files = [
            "e:/github_project/d2l_note/runs/run_20250914_044631/metrics.json",
            "e:/github_project/d2l_note/runs/run_20250914_040635/metrics.json"  # 尝试另一个可能存在的指标文件
        ]
        
        # 过滤掉不存在的文件
        self.existing_metrics_files = [
            file_path for file_path in self.test_metrics_files 
            if os.path.exists(file_path)
        ]
        
    def test_parse_metrics_file(self):
        """测试解析指标文件的功能"""
        # 如果没有找到测试文件，跳过测试
        if not self.existing_metrics_files:
            self.skipTest(f"未找到测试用的指标文件: {', '.join(self.test_metrics_files)}")
            return
        
        for file_path in self.existing_metrics_files:
            print(f"\n测试解析文件: {file_path}")
            
            # 使用便捷函数解析文件
            model_info = parse_model_info(
                file_path=file_path,
                namespace="metrics"
            )
            
            # 验证解析结果
            self.assertIsNotNone(model_info, f"解析文件失败: {file_path}")
            self.assertIsInstance(model_info, ModelInfoData, "解析结果不是ModelInfoData类型")
            
            # 验证必要字段是否存在
            self.assertTrue(hasattr(model_info, 'name'), "缺少name字段")
            self.assertTrue(hasattr(model_info, 'path'), "缺少path字段")
            self.assertTrue(hasattr(model_info, 'timestamp'), "缺少timestamp字段")
            self.assertTrue(hasattr(model_info, 'metrics'), "缺少metrics字段")
            
            # 验证指标数据
            metrics = model_info.metrics
            self.assertIn('final_test_acc', metrics, "metrics中缺少final_test_acc字段")
            self.assertIn('best_test_acc', metrics, "metrics中缺少best_test_acc字段")
            self.assertIn('final_train_loss', metrics, "metrics中缺少final_train_loss字段")
            self.assertIn('total_training_time', metrics, "metrics中缺少total_training_time字段")
            
            # 验证指标值的类型
            self.assertIsInstance(metrics['final_test_acc'], (int, float), "final_test_acc不是数字类型")
            self.assertIsInstance(metrics['best_test_acc'], (int, float), "best_test_acc不是数字类型")
            self.assertIsInstance(metrics['final_train_loss'], (int, float), "final_train_loss不是数字类型")
            self.assertIsInstance(metrics['total_training_time'], str, "total_training_time不是字符串类型")
            
            # 打印解析结果（用于调试）
            print(f"解析成功: {model_info.name}")
            print(f"  - 最终测试准确率: {metrics['final_test_acc']:.4f}")
            print(f"  - 最佳测试准确率: {metrics['best_test_acc']:.4f}")
            print(f"  - 最终训练损失: {metrics['final_train_loss']:.6f}")
            print(f"  - 总训练时间: {metrics['total_training_time']}")
            print(f"  - 模型类型: {model_info.model_type}")
            print(f"  - 框架: {model_info.framework}")
            print(f"  - 任务类型: {model_info.task_type}")
    
    def test_multiple_files(self):
        """测试解析多个指标文件的功能"""
        # 如果没有找到测试文件，跳过测试
        if not self.existing_metrics_files:
            self.skipTest(f"未找到测试用的指标文件: {', '.join(self.test_metrics_files)}")
            return
            
        # 解析所有找到的文件并比较结果
        results = []
        for file_path in self.existing_metrics_files:
            model_info = parse_model_info(file_path=file_path, namespace="metrics")
            if model_info:
                results.append((file_path, model_info))
        
        # 验证至少解析了一个文件
        self.assertTrue(len(results) > 0, "未能解析任何指标文件")
        
        # 如果解析了多个文件，比较它们的差异
        if len(results) > 1:
            print(f"\n比较 {len(results)} 个解析结果:")
            first_path, first_result = results[0]
            for path, result in results[1:]:
                print(f"\n比较 {os.path.basename(first_path)} 和 {os.path.basename(path)}:")
                print(f"  准确率差异: {abs(result.metrics.get('final_test_acc', 0) - first_result.metrics.get('final_test_acc', 0)):.4f}")
                print(f"  损失值差异: {abs(result.metrics.get('final_train_loss', 0) - first_result.metrics.get('final_train_loss', 0)):.6f}")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main()