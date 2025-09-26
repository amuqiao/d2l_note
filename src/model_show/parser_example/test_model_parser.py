"""
模型文件解析器测试用例

功能：测试ModelFileParser解析PyTorch模型文件(.pth)的能力，验证其能否正确提取模型名称、类型、大小等信息。
使用示例：python -m unittest src.model_show.parser_example.test_model_parser
"""
import unittest
import os
from src.model_show.parsers.base_parsers import ModelInfoParserRegistry
from src.model_show.data_models import ModelInfoData


class TestModelParser(unittest.TestCase):
    """模型文件解析器测试用例"""
    
    def setUp(self):
        """设置测试环境，准备测试文件路径"""
        # 测试的模型文件路径
        self.model_file = "e:/github_project/d2l_note/runs/run_20250914_040635/best_model_LeNet_acc_0.7860_epoch_9.pth"
        
        # 验证测试文件是否存在
        if not os.path.exists(self.model_file):
            self.skipTest(f"测试文件不存在: {self.model_file}")
    
    def test_parse_model_file(self):
        """测试模型文件解析功能"""
        # 使用解析器注册表解析模型文件
        model_info = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file,
            namespace="models"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info, ModelInfoData)
        self.assertIn("best_model_LeNet_acc_0.7860_epoch_9", model_info.name)
        self.assertEqual(model_info.path, self.model_file)
        self.assertEqual(model_info.model_type, "PyTorch")
        self.assertEqual(model_info.namespace, "models")
        
        # 验证params字典是否正确创建
        self.assertIsInstance(model_info.params, dict)
        self.assertIn("model_size", model_info.params)
        self.assertIsNotNone(model_info.params["model_size"])
        self.assertIn("file_extension", model_info.params)
        self.assertEqual(model_info.params["file_extension"], ".pth")
        
        # 打印解析结果
        print(f"\n模型文件解析结果:")
        print(f"模型名称: {model_info.name}")
        print(f"模型路径: {model_info.path}")
        print(f"模型类型: {model_info.model_type}")
        print(f"命名空间: {model_info.namespace}")
        print(f"时间戳: {model_info.timestamp}")
        print(f"参数字典: {model_info.params}")
        
    def test_extract_model_info_from_filename(self):
        """测试从文件名中提取模型信息"""
        # 使用解析器注册表解析模型文件
        model_info = ModelInfoParserRegistry.parse_file(
            file_path=self.model_file,
            namespace="models"
        )
        
        # 从文件名提取信息
        file_name = os.path.basename(self.model_file)
        
        # 验证模型名称中包含LeNet
        self.assertIn("LeNet", model_info.name)
        
        # 验证文件名中包含准确率信息
        self.assertIn("0.7860", file_name)
        
        # 验证文件名中包含epoch信息
        self.assertIn("epoch_9", file_name)
        
        print(f"\n从文件名提取的信息:")
        print(f"文件名: {file_name}")
        print(f"检测到的模型名称: LeNet")
        print(f"检测到的准确率: 0.7860")
        print(f"检测到的训练轮次: 9")


if __name__ == "__main__":
    unittest.main()