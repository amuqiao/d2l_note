"""
配置文件可视化器测试用例

功能：测试ConfigFileVisualizer可视化配置文件解析结果的能力，验证其能否正确生成配置概览、详情和可视化元素。
使用示例：python -m unittest src.model_show.visualizer_example.test_config_visualizer
"""
import unittest
import os
from src.model_show.parser_registry import parse_model_info
from src.model_show.visualizer_registry import visualize_model_info
from src.model_show.visualizers.base_model_visualizers import ModelVisualizerRegistry
from src.model_show.data_models import ModelInfoData
from typing import Dict, Any


class TestConfigVisualizer(unittest.TestCase):
    """配置文件可视化器测试用例"""
    
    def setUp(self):
        """设置测试环境，准备测试文件路径"""
        # 测试的配置文件路径
        self.config_file1 = "e:/github_project/d2l_note/runs/run_20250914_040635/config.json"
        self.config_file2 = "e:/github_project/d2l_note/runs/run_20250914_044631/config.json"
        
        # 验证测试文件是否存在
        for file_path in [self.config_file1, self.config_file2]:
            if not os.path.exists(file_path):
                self.skipTest(f"测试文件不存在: {file_path}")
        
        # 先解析配置文件，获取模型信息数据
        self.model_info1 = parse_model_info(self.config_file1, namespace="config")
        self.model_info2 = parse_model_info(self.config_file2, namespace="config")
        
        # 验证解析结果是否有效
        if not self.model_info1 or not self.model_info2:
            self.skipTest("配置文件解析失败，无法进行可视化测试")
    
    def test_visualize_config(self):
        """测试配置文件可视化功能"""
        # 使用可视化器注册中心可视化第一个配置
        visualization1 = ModelVisualizerRegistry.visualize_model(
            model_info=self.model_info1,
            namespace="config"
        )
        
        # 使用便捷函数可视化第二个配置
        visualization2 = visualize_model_info(self.model_info2, namespace="config")
        
        # 验证可视化结果
        self.assertIsNotNone(visualization1)
        self.assertIsNotNone(visualization2)
        self.assertIsInstance(visualization1, dict)
        self.assertIsInstance(visualization2, dict)
        
        # 验证可视化结果的基本结构
        self.assertIn("type", visualization1)
        self.assertEqual(visualization1["type"], "config_visualization")
        self.assertIn("model_name", visualization1)
        self.assertEqual(visualization1["model_name"], self.model_info1.name)
        
        # 验证配置概览信息
        self.assertIn("config_overview", visualization1)
        self.assertIsInstance(visualization1["config_overview"], dict)
        self.assertIn("parameter_count", visualization1["config_overview"])
        
        # 验证配置详情
        self.assertIn("config_details", visualization1)
        self.assertIsInstance(visualization1["config_details"], dict)
        
        # 验证可视化元素
        self.assertIn("visualization_elements", visualization1)
        self.assertIsInstance(visualization1["visualization_elements"], dict)
        self.assertIn("config_tree", visualization1["visualization_elements"])
        self.assertIn("param_distribution", visualization1["visualization_elements"])
        
        print(f"\n配置文件1可视化结果结构:")
        print(f"类型: {visualization1.get('type')}")
        print(f"模型名称: {visualization1.get('model_name')}")
        print(f"配置参数数量: {visualization1.get('config_overview', {}).get('parameter_count')}")
        print(f"关键参数数量: {len(visualization1.get('config_overview', {}).get('key_parameters', []))}")
        print(f"可视化元素: {list(visualization1.get('visualization_elements', {}).keys())}")
        
        print(f"\n配置文件2可视化结果结构:")
        print(f"类型: {visualization2.get('type')}")
        print(f"模型名称: {visualization2.get('model_name')}")
        print(f"配置参数数量: {visualization2.get('config_overview', {}).get('parameter_count')}")
        print(f"关键参数数量: {len(visualization2.get('config_overview', {}).get('key_parameters', []))}")
        print(f"可视化元素: {list(visualization2.get('visualization_elements', {}).keys())}")
    
    def test_visualizer_support(self):
        """测试可视化器的支持函数"""
        # 从注册中心获取所有可视化器
        from src.model_show.visualizers.config_file_visualizer import ConfigFileVisualizer
        
        # 创建可视化器实例
        visualizer = ConfigFileVisualizer()
        
        # 测试支持函数
        self.assertTrue(visualizer.support(self.model_info1, "config"))
        self.assertTrue(visualizer.support(self.model_info2, "config"))
        
        # 修改命名空间，测试是否仍能支持
        self.model_info1.namespace = "test"
        self.assertTrue(visualizer.support(self.model_info1, "test"))  # 应该仍能支持，因为文件路径包含config
        
        print(f"\n可视化器支持函数测试通过")
    
    def test_namespace_support(self):
        """测试命名空间支持"""
        # 使用不同的命名空间测试
        visualization_default = ModelVisualizerRegistry.visualize_model(
            model_info=self.model_info1,
            namespace="default"
        )
        
        visualization_config = ModelVisualizerRegistry.visualize_model(
            model_info=self.model_info1,
            namespace="config"
        )
        
        # 验证两种命名空间都能正常工作
        self.assertIsNotNone(visualization_default)
        self.assertIsNotNone(visualization_config)
        
        print(f"\n命名空间测试通过：")
        print(f"default命名空间可视化结果类型: {visualization_default.get('type')}")
        print(f"config命名空间可视化结果类型: {visualization_config.get('type')}")


if __name__ == "__main__":
    unittest.main()