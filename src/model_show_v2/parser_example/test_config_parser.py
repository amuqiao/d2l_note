"""
配置文件解析器测试用例

功能：测试ConfigFileParser解析config.json文件的能力，验证其能否正确提取模型配置参数、学习率、批量大小等信息，并比较多个配置文件的差异。
使用示例：python -m unittest src.model_show_v2.parser_example.test_config_parser
"""
import unittest
import os
from src.model_show_v2.parsers.base_model_parsers import ModelInfoParserRegistry
from src.model_show_v2.data_models import ModelInfoData


class TestConfigParser(unittest.TestCase):
    """配置文件解析器测试用例"""
    
    def setUp(self):
        """设置测试环境，准备测试文件路径"""
        # 测试的配置文件路径
        self.config_file1 = "e:/github_project/d2l_note/runs/run_20250914_040635/config.json"
        self.config_file2 = "e:/github_project/d2l_note/runs/run_20250927_235759/config.json"
        
        # 验证测试文件是否存在
        for file_path in [self.config_file1, self.config_file2]:
            if not os.path.exists(file_path):
                self.skipTest(f"测试文件不存在: {file_path}")
    
    def test_parse_config_file(self):
        """测试配置文件解析功能"""
        # 解析第一个配置文件
        model_info1 = ModelInfoParserRegistry.parse_file(
            file_path=self.config_file1,
            namespace="config"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info1, ModelInfoData)
        self.assertEqual(model_info1.name, "LeNet")
        self.assertEqual(model_info1.path, self.config_file1)
        self.assertEqual(model_info1.model_type, "LeNet")
        
        # 验证参数是否正确解析
        self.assertIn("model_name", model_info1.params)
        self.assertEqual(model_info1.params["model_name"], "LeNet")
        self.assertIn("num_epochs", model_info1.params)
        self.assertEqual(model_info1.params["num_epochs"], 10)
        self.assertIn("learning_rate", model_info1.params)
        self.assertEqual(model_info1.params["learning_rate"], 0.8)
        
        # 解析第二个配置文件
        model_info2 = ModelInfoParserRegistry.parse_file(
            file_path=self.config_file2,
            namespace="config"
        )
        
        # 验证解析结果
        self.assertIsInstance(model_info2, ModelInfoData)
        self.assertEqual(model_info2.name, "LeNet")
        self.assertEqual(model_info2.path, self.config_file2)
        self.assertEqual(model_info2.model_type, "LeNet")
        
        # 验证参数是否正确解析
        self.assertIn("model_name", model_info2.params)
        self.assertEqual(model_info2.params["model_name"], "LeNet")
        self.assertIn("num_epochs", model_info2.params)
        self.assertEqual(model_info2.params["num_epochs"], 30)
        self.assertIn("learning_rate", model_info2.params)
        self.assertEqual(model_info2.params["learning_rate"], 0.8)
        
        print(f"\n配置文件1解析结果:")
        print(f"模型名称: {model_info1.name}")
        print(f"模型类型: {model_info1.model_type}")
        print(f"参数字典: {model_info1.params}")
        print(f"框架: {model_info1.framework}")
        print(f"任务类型: {model_info1.task_type}")
        print(f"版本: {model_info1.version}")
        print(f"时间戳: {model_info1.timestamp}")
        print(f"指标数量: {len(model_info1.metric_list)}")
        
        print(f"\n配置文件2解析结果:")
        print(f"模型名称: {model_info2.name}")
        print(f"模型类型: {model_info2.model_type}")
        print(f"参数字典: {model_info2.params}")
        print(f"框架: {model_info2.framework}")
        print(f"任务类型: {model_info2.task_type}")
        print(f"版本: {model_info2.version}")
        print(f"时间戳: {model_info2.timestamp}")
        print(f"指标数量: {len(model_info2.metric_list)}")
    
    def test_multiple_config_files_comparison(self):
        """测试多个配置文件的解析和比较"""
        # 解析两个配置文件
        model_info1 = ModelInfoParserRegistry.parse_file(
            file_path=self.config_file1,
            namespace="config"
        )
        model_info2 = ModelInfoParserRegistry.parse_file(
            file_path=self.config_file2,
            namespace="config"
        )
        
        # 比较两个配置的差异
        self.assertEqual(model_info1.name, model_info2.name)
        self.assertEqual(model_info1.model_type, model_info2.model_type)
        self.assertNotEqual(model_info1.path, model_info2.path)
        self.assertNotEqual(model_info1.timestamp, model_info2.timestamp)
        self.assertNotEqual(model_info1.params["num_epochs"], model_info2.params["num_epochs"])
        
        # 比较学习率差异
        lr1 = model_info1.params.get("learning_rate", 0)
        lr2 = model_info2.params.get("learning_rate", 0)
        lr_diff = abs(lr1 - lr2)
        
        # 比较批量大小差异
        bs1 = model_info1.params.get("batch_size", 0)
        bs2 = model_info2.params.get("batch_size", 0)
        bs_diff = abs(bs1 - bs2)
        
        print(f"\n配置文件比较结果:")
        print(f"模型名称差异: {model_info1.name} vs {model_info2.name}")
        print(f"学习率差异: {lr1} vs {lr2} (差值: {lr_diff})")
        print(f"批量大小差异: {bs1} vs {bs2} (差值: {bs_diff})")
        print(f"设备相同: {model_info1.params.get('device') == model_info2.params.get('device')}")
        print(f"训练轮数相同: {model_info1.params.get('num_epochs') == model_info2.params.get('num_epochs')}")


if __name__ == "__main__":
    unittest.main()