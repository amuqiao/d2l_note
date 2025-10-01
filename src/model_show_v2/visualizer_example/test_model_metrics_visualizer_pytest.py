"""
模型指标可视化器测试：验证MetricsVisualizer的功能（pytest版本）

功能：测试模型训练指标的可视化和比较功能
"""
import logging
import os
from typing import Dict, Any, List

import pytest

from src.model_show_v2.parsers.model_metrics_parser import MetricsFileParser
from src.model_show_v2.visualizers.model_metrics_visualizer import MetricsVisualizer
from src.model_show_v2.data_models import ModelInfoData
from src.utils.log_utils import get_logger

# 设置日志级别为ERROR，减少测试输出
logger = get_logger()
logger.set_global_level(logging.ERROR)

# 准备测试数据的fixture
@pytest.fixture
def test_data_paths():
    """准备测试数据路径"""
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
        pytest.fail(f"测试文件不存在: {metrics_file_path1}")
    
    if not os.path.exists(metrics_file_path2):
        pytest.fail(f"测试文件不存在: {metrics_file_path2}")
    
    return metrics_file_path1, metrics_file_path2

# 准备模型信息的fixture
@pytest.fixture
def model_infos(test_data_paths):
    """准备模型信息数据"""
    parser = MetricsFileParser()
    metrics_file_path1, metrics_file_path2 = test_data_paths
    
    # 解析指标文件，获取模型信息
    model_info1 = parser.parse(metrics_file_path1)
    model_info2 = parser.parse(metrics_file_path2)
    
    return model_info1, model_info2

# 准备可视化器的fixture
@pytest.fixture
def visualizer():
    """准备可视化器实例"""
    return MetricsVisualizer()

# 准备解析器的fixture
@pytest.fixture
def parser():
    """准备解析器实例"""
    return MetricsFileParser()

# 测试可视化功能
def test_visualize(model_infos, visualizer, request):
    """测试可视化功能"""
    model_info, _ = model_infos
    
    # 执行可视化
    result = visualizer.visualize(model_info)
    
    # 验证可视化结果
    assert result["success"], f"可视化失败: {result.get('message')}"
    assert "text" in result, "可视化结果中缺少text字段"
    assert "tables" in result, "可视化结果中缺少tables字段"
    
    # 验证生成的表格类型
    tables = result["tables"]
    assert "optimized_metrics" in tables, "缺少优化的指标表格"
    
    # 验证表格内容是否包含关键信息
    text = result["text"]
    assert model_info.name in text, "可视化文本中未包含模型名称"
    
    # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
    test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
    assert test_acc_related, "可视化文本中未包含测试准确率相关信息"
    
    # 验证表格内容是否包含关键信息
    optimized_table = tables["optimized_metrics"]
    # 验证表格中包含模型名称
    assert model_info.name in str(optimized_table), "表格应包含模型名称"
    # 验证表格中包含关键指标字段
    table_str = str(optimized_table).lower()
    assert "final_train_loss" in table_str, "表格应包含最终训练损失"
    assert "final_test_acc" in table_str, "表格应包含最终测试准确率"
    assert "best_test_acc" in table_str, "表格应包含最佳测试准确率"
    assert "total_training_time" in table_str, "表格应包含总训练时间"
    
    # 显示可视化结果
    print("\n=== 可视化测试结果 ===")
    print(result["text"])
    print("=== 可视化测试完成 ===\n")

# 测试支持判断功能
def test_support(model_infos, visualizer):
    """测试支持判断功能"""
    model_info, _ = model_infos
    
    # 测试支持判断
    is_supported = visualizer.support(model_info)
    
    # 验证支持判断结果
    assert is_supported, "应该支持该模型指标信息"
    
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
    is_supported_empty = visualizer.support(empty_model_info)
    assert not is_supported_empty, "不应该支持没有指标数据的模型信息"

# 测试比较功能
def test_compare(model_infos, visualizer):
    """测试比较功能"""
    model_info1, model_info2 = model_infos
    
    # 创建模型信息列表用于比较
    model_infos_list = [model_info1, model_info2]
    
    # 执行比较
    result = visualizer.compare(model_infos_list)
    
    # 验证比较结果
    assert result["success"], f"比较失败: {result.get('message')}"
    assert "text" in result, "比较结果中缺少text字段"
    assert "tables" in result, "比较结果中缺少tables字段"
    
    # 验证生成的表格类型
    tables = result["tables"]
    assert "optimized_compare" in tables, "缺少优化的比较表格"
    
    # 验证表格内容是否包含关键信息
    text = result["text"]
    assert model_info1.name in text, "比较文本中未包含第一个模型名称"
    assert model_info2.name in text, "比较文本中未包含第二个模型名称"
    
    # 检查是否包含测试准确率相关信息（可能有不同的文本表示）
    test_acc_related = any(keyword in text.lower() for keyword in ["test", "准确率"])
    assert test_acc_related, "比较文本中未包含测试准确率相关信息"
    
    # 验证表格内容是否包含关键信息
    compare_table = tables["optimized_compare"]
    # 验证表格中包含两个模型的名称
    assert model_info1.name in str(compare_table), "表格应包含第一个模型名称"
    assert model_info2.name in str(compare_table), "表格应包含第二个模型名称"
    # 验证表格中包含关键指标字段
    table_str = str(compare_table).lower()
    assert "final_train_loss" in table_str, "表格应包含最终训练损失"
    assert "final_test_acc" in table_str, "表格应包含最终测试准确率"
    assert "best_test_acc" in table_str, "表格应包含最佳测试准确率"
    assert "total_training_time" in table_str, "表格应包含总训练时间"
    
    # 显示比较结果
    print("\n=== 比较测试结果 ===")
    print(result["text"])
    print("=== 比较测试完成 ===\n")

# 测试比较单个模型时的错误处理
def test_compare_single_model(model_infos, visualizer):
    """测试比较单个模型时的错误处理"""
    model_info, _ = model_infos
    
    # 创建只有一个模型信息的列表
    model_infos_list = [model_info]
    
    # 执行比较
    result = visualizer.compare(model_infos_list)
    
    # 验证比较结果（应该失败）
    assert not result["success"], "比较单个模型时应该失败"
    assert "需要至少2个模型指标信息" in result.get("message", ""), "错误信息不正确"

# 测试可视化器与注册中心的集成
def test_visualizer_registry_integration(model_infos):
    """测试可视化器与注册中心的集成"""
    model_info, _ = model_infos
    
    from src.model_show_v2.visualizers.base_model_visualizers import ModelVisualizerRegistry
    
    # 检查可视化器是否已注册
    registry = ModelVisualizerRegistry
    visualizer = registry.get_matched_visualizer(model_info)
    
    # 验证找到的可视化器是否是MetricsVisualizer的实例
    assert isinstance(visualizer, MetricsVisualizer), "未能从注册中心找到正确的可视化器"
    
    # 直接使用注册中心的visualize_model方法进行测试
    result = registry.visualize_model(model_info)
    assert result["success"], "通过注册中心进行可视化失败"
    
    # 显示通过注册中心的可视化结果
    print("\n=== 注册中心可视化测试结果 ===")
    print(result["text"] if result.get("text") else "可视化成功")
    print("=== 注册中心可视化测试完成 ===\n")

# 展示模型指标可视化效果
def show_visualization_effect():
    """展示模型指标可视化效果"""
    print("\n=== 模型指标可视化效果展示 ===")
    print("此脚本可以直接运行查看可视化效果，也可以作为pytest单元测试运行")
    print("\npytest的优势：")
    print("1. 提供更丰富的功能和更简洁的语法")
    print("2. 支持fixture机制进行测试数据管理")
    print("3. 提供自动发现测试、更丰富的断言、参数化测试等高级功能")
    print("4. 可以生成详细的测试报告")
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

if __name__ == "__main__":
    # 如果直接运行脚本，展示可视化效果
    show_visualization_effect()
    # 如果需要运行pytest测试，可以使用命令行：pytest -v test_model_metrics_visualizer_pytest.py