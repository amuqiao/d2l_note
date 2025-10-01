"""
模型文件解析器测试用例

功能：测试ModelFileParser解析模型文件的能力，验证其能否正确提取模型结构、参数数量等信息，并比较多个模型文件的差异。
使用示例：python -m unittest src.model_show_v2.parser_example.test_model_parser
"""
import unittest
import os
from src.model_show_v2.parsers.base_model_parsers import ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData, MetricData


class TestModelFileParser(unittest.TestCase):
    """模型文件解析器测试用例"""
    
    def setUp(self):
        """设置测试环境，准备测试文件路径"""
        # 测试的模型文件路径（根据实际情况修改）
        self.model_file1 = "e:/github_project/d2l_note/runs/run_20250914_040635/best_model_LeNet_acc_0.7860_epoch_9.pth"
        self.model_file2 = "e:/github_project/d2l_note/runs/run_20250914_040809/best_model_AlexNet_acc_0.9107_epoch_28.pth"
        
        # 验证测试文件是否存在
        for file_path in [self.model_file1, self.model_file2]:
            if not os.path.exists(file_path):
                self.skipTest(f"测试文件不存在: {file_path}")
    
    def test_parse_model_file(self):
        """测试模型文件解析功能"""
        # 解析第一个模型文件
        model_info1 = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file1,
            namespace="models"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info1, ModelInfoData)
        self.assertIsNotNone(model_info1.name)
        self.assertEqual(model_info1.path, self.model_file1)
        self.assertIn(model_info1.model_type, ["PyTorch", "unknown"])
        
        # 验证参数是否正确解析
        self.assertIn("model_size", model_info1.params)
        self.assertGreater(model_info1.params["model_size"], 0)
        
        # 验证指标是否正确添加
        self.assertGreater(len(model_info1.metric_list), 0)
        param_metrics = [m for m in model_info1.metric_list if m.name == "Parameters"]
        size_metrics = [m for m in model_info1.metric_list if m.name == "Model Size"]
        
        if param_metrics:
            self.assertGreaterEqual(param_metrics[0].data.get('value', 0), 0)
        if size_metrics:
            self.assertGreater(size_metrics[0].data.get('value', 0), 0)
            self.assertEqual(size_metrics[0].data.get('unit', ''), "MB")
        
        # 解析第二个模型文件
        model_info2 = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file2,
            namespace="models"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info2, ModelInfoData)
        self.assertIsNotNone(model_info2.name)
        self.assertEqual(model_info2.path, self.model_file2)
        self.assertIn(model_info2.model_type, ["PyTorch", "unknown"])
        
        # 验证参数是否正确解析
        self.assertIn("model_size", model_info2.params)
        self.assertGreater(model_info2.params["model_size"], 0)
        
        # 验证指标是否正确添加
        self.assertGreater(len(model_info2.metric_list), 0)
        
        # 验证第二个模型的指标
        param_metrics2 = [m for m in model_info2.metric_list if m.name == "Parameters"]
        size_metrics2 = [m for m in model_info2.metric_list if m.name == "Model Size"]
        
        if param_metrics2:
            self.assertGreaterEqual(param_metrics2[0].data.get('value', 0), 0)
        if size_metrics2:
            self.assertGreater(size_metrics2[0].data.get('value', 0), 0)
            self.assertEqual(size_metrics2[0].data.get('unit', ''), "MB")
        
        print(f"\n模型文件1解析结果:")
        print(f"模型名称: {model_info1.name}")
        print(f"模型类型: {model_info1.model_type}")
        print(f"参数字典: {model_info1.params}")
        print(f"框架: {model_info1.framework}")
        print(f"任务类型: {model_info1.task_type}")
        print(f"版本: {model_info1.version}")
        print(f"时间戳: {model_info1.timestamp}")
        print(f"指标数量: {len(model_info1.metric_list)}")
        
        # 打印指标详情
        for metric in model_info1.metric_list:
            print(f"  - {metric.name}: {metric.data.get('value', '-')}{metric.data.get('unit', '')} ({metric.description})")
        
        print(f"\n模型文件2解析结果:")
        print(f"模型名称: {model_info2.name}")
        print(f"模型类型: {model_info2.model_type}")
        print(f"参数字典: {model_info2.params}")
        print(f"框架: {model_info2.framework}")
        print(f"任务类型: {model_info2.task_type}")
        print(f"版本: {model_info2.version}")
        print(f"时间戳: {model_info2.timestamp}")
        print(f"指标数量: {len(model_info2.metric_list)}")
        
        # 打印指标详情
        for metric in model_info2.metric_list:
            print(f"  - {metric.name}: {metric.data.get('value', '-')}{metric.data.get('unit', '')} ({metric.description})")
    
    def test_multiple_model_files_comparison(self):
        """测试多个模型文件的解析和比较"""
        # 解析两个模型文件
        model_info1 = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file1,
            namespace="models"
        )
        model_info2 = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file2,
            namespace="models"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info1, ModelInfoData)
        self.assertIsInstance(model_info2, ModelInfoData)
        
        # 比较两个模型文件的基本信息
        print(f"\n模型文件比较:")
        print(f"文件名比较: {model_info1.name} vs {model_info2.name}")
        print(f"文件大小比较: {model_info1.params.get('model_size', 0)}MB vs {model_info2.params.get('model_size', 0)}MB")
        
        # 获取参数数量指标
        params1 = None
        params2 = None
        for metric in model_info1.metric_list:
            if metric.name == "Parameters" and metric.data.get('value') is not None:
                params1 = metric.data.get('value')
                break
        for metric in model_info2.metric_list:
            if metric.name == "Parameters" and metric.data.get('value') is not None:
                params2 = metric.data.get('value')
                break
        
        if params1 is not None and params2 is not None:
            print(f"参数数量比较: {params1} vs {params2}")
            print(f"参数数量差异: {abs(params1 - params2)} ({abs(params1 - params2) / max(params1, params2) * 100:.2f}%)")
    

if __name__ == "__main__":
    unittest.main()