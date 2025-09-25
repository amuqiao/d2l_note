import os
import sys
import time
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_visualization.model_info_parsers import ModelInfoParserRegistry, register_parsers
from src.model_visualization.model_info_visualizers_impl import ModelComparisonVisualizer

# 注册解析器
register_parsers()

def create_test_model_info():
    """创建测试用的模型信息数据"""
    # 模拟创建几个模型信息对象用于测试
    from src.model_visualization.data_models import ModelInfoData
    
    # 获取当前时间戳
    current_timestamp = time.time()
    
    # 模型1
    model1 = ModelInfoData(
        type="metrics",
        path="runs/test_model1",
        model_type="ResNet-18",
        params={"total_params": 11689512, "model_type": "ResNet-18"},
        metrics={
            "final_test_acc": 0.9256,
            "final_train_acc": 0.9532,
            "best_val_acc": 0.9312,
            "total_training_time": "1234.5 seconds"
        },
        timestamp=current_timestamp - 3600,  # 1小时前
        namespace="test"
    )
    
    # 模型2
    model2 = ModelInfoData(
        type="metrics",
        path="runs/test_model2",
        model_type="VGG-16",
        params={"total_params": 138357544, "model_type": "VGG-16"},
        metrics={
            "final_test_acc": 0.9421,
            "final_train_acc": 0.9876,
            "best_val_acc": 0.9513,
            "total_training_time": "2345.6 seconds"
        },
        timestamp=current_timestamp - 7200,  # 2小时前
        namespace="test"
    )
    
    # 模型3
    model3 = ModelInfoData(
        type="metrics",
        path="runs/test_model3",
        model_type="AlexNet",
        params={"total_params": 61100840, "model_type": "AlexNet"},
        metrics={
            "final_test_acc": 0.8975,
            "final_train_acc": 0.9234,
            "best_val_acc": 0.9012,
            "total_training_time": "876.5 seconds"
        },
        timestamp=current_timestamp - 10800,  # 3小时前
        namespace="test"
    )
    
    return [model1, model2, model3]

def main():
    """主测试函数"""
    print("="*80)
    print("🧪 模型比较表格输出测试")
    print("="*80)
    
    # 创建测试数据
    model_infos = create_test_model_info()
    
    # 创建比较可视化器
    visualizer = ModelComparisonVisualizer()
    
    # 添加模型信息
    for model_info in model_infos:
        visualizer.add_model_info(model_info)
    
    # 设置按准确率排序
    visualizer.set_sort_by("accuracy")
    
    # 使用ranking模式进行可视化（表格输出）
    print("\n📊 测试ranking模式（按准确率排序）:")
    table = visualizer.visualize(show=True, plot_type="ranking")
    
    # 测试按名称排序
    visualizer = ModelComparisonVisualizer()  # 创建新的可视化器
    for model_info in model_infos:
        visualizer.add_model_info(model_info)
    visualizer.set_sort_by("name")
    
    print("\n📊 测试ranking模式（按名称排序）:")
    table = visualizer.visualize(show=True, plot_type="ranking")
    
    # 测试其他模式（应该显示警告）
    print("\n📊 测试其他模式（应该显示警告）:")
    table = visualizer.visualize(show=True, plot_type="bar")
    
    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)

if __name__ == "__main__":
    main()