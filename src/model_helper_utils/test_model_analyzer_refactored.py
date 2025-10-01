#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试重构后的模型分析工具是否能正常工作"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock

# 确保可以导入model_analyzer_refactored
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入重构后的模块
from src.model_helper_utils.model_analyzer_refactored import (
    ModelAnalysisService, PathScanner, MetricParserRegistry, 
    VisualizerRegistry, DataAccessor, ModelInfoData, MetricData
)


class TestModelAnalyzerRefactored(unittest.TestCase):
    """测试重构后的模型分析工具"""
    
    def setUp(self):
        """设置测试环境"""
        # 禁用日志输出，避免干扰测试
        logging.disable(logging.CRITICAL)
        
        # 模拟一些测试路径
        self.test_run_dir = "runs/run_test"
        self.test_metrics_path = "runs/run_test/metrics.json"
        
    def tearDown(self):
        """清理测试环境"""
        # 恢复日志设置
        logging.disable(logging.NOTSET)
    
    @patch('glob.glob')
    def test_path_scanner_find_run_directories(self, mock_glob):
        """测试PathScanner.find_run_directories方法"""
        # 配置mock行为 - 根据实际方法的实现调整
        mock_glob.return_value = ['./runs/run_1', './runs/run_2', './runs/config.json']
        
        # 调用被测方法
        result = PathScanner.find_run_directories(pattern="run_*", root_dir="./runs")
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertIn('./runs/run_1', result)
        self.assertIn('./runs/run_2', result)
    
    @patch('src.model_helper_utils.model_analyzer_refactored.DataAccessor.read_file')
    def test_model_data_processor_extract_run_metrics(self, mock_read_file):
        """测试ModelDataProcessor.extract_run_metrics方法"""
        # 配置mock行为
        mock_read_file.return_value = {
            'best_test_acc': 0.95,
            'final_test_acc': 0.94,
            'total_training_time': '10m 30s'
        }
        
        # 导入ModelDataProcessor
        from src.model_helper_utils.model_analyzer_refactored import ModelDataProcessor
        
        # 调用被测方法
        metrics = ModelDataProcessor.extract_run_metrics(self.test_run_dir)
        
        # 验证结果
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics['best_acc'], 0.95)
        self.assertEqual(metrics['final_acc'], 0.94)
        self.assertEqual(metrics['time_cost'], '10m 30s')
    
    @patch('src.model_helper_utils.model_analyzer_refactored.PathScanner.find_run_directories')
    @patch('src.model_helper_utils.model_analyzer_refactored.ModelDataProcessor.create_run_info')
    @patch('src.model_helper_utils.model_analyzer_refactored.ResultVisualizer.print_summary_table')
    @patch('src.model_helper_utils.model_analyzer_refactored.ResultVisualizer.print_statistics')
    def test_model_analysis_service_summarize_runs(self, 
                                                   mock_print_stats, 
                                                   mock_print_table, 
                                                   mock_create_info, 
                                                   mock_find_dirs):
        """测试ModelAnalysisService.summarize_runs方法"""
        # 配置mock行为
        mock_find_dirs.return_value = ['runs/run_1', 'runs/run_2']
        
        # 创建模拟的ModelInfoData对象
        mock_info = MagicMock(spec=ModelInfoData)
        mock_info.type = "run"
        mock_info.metrics = {'best_acc': 0.95}
        mock_create_info.return_value = mock_info
        
        # 调用被测方法
        result = ModelAnalysisService.summarize_runs(run_dir_pattern="run_\*", top_n=10, root_dir="./runs")
        
        # 验证结果
        self.assertEqual(len(result), 2)
        mock_print_table.assert_called_once()
        mock_print_stats.assert_called_once()
    
    def test_data_models_initialization(self):
        """测试数据模型的初始化"""
        # 测试MetricData
        metric_data = MetricData(
            metric_type="epoch_curve",
            data={'epochs': [1, 2, 3], 'train_loss': [0.5, 0.3, 0.1]},
            source_path="test/metrics.json",
            timestamp=1234567890.0
        )
        self.assertEqual(metric_data.metric_type, "epoch_curve")
        self.assertEqual(len(metric_data.data['epochs']), 3)
        
        # 测试ModelInfoData
        model_info = ModelInfoData(
            type="run",
            path="test/run_dir",
            model_type="CNN",
            params={'lr': 0.001, 'batch_size': 32},
            metrics={'best_acc': 0.95},
            timestamp=1234567890.0
        )
        self.assertEqual(model_info.type, "run")
        self.assertEqual(model_info.model_type, "CNN")
        

if __name__ == '__main__':
    unittest.main()