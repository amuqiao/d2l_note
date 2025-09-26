import os
from src.model_visualization.model_info_parsers import ModelInfoParserRegistry
from src.model_visualization.model_info_visualizers import ModelInfoVisualizerRegistry, ModelComparisonVisualizer


def main():
    # 示例文件路径
    run_dir = "runs/run_20250914_040635"
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    model_path = os.path.join(run_dir, "best_model_LeNet_acc_0.7860_epoch_9.pth")
    
    print("=" * 80)
    print("ModelInfoData解析器和可视化器演示")
    print("=" * 80)
    
    # 解析并可视化config.json
    print("\n1. 解析并可视化配置文件 (config.json):")
    config_info = ModelInfoParserRegistry.parse_file(config_path)
    if config_info:
        print(f"\n解析成功! 模型类型: {config_info.model_type}")
        ModelInfoVisualizerRegistry.draw(config_info)
    else:
        print("解析失败!")
    
    print("\n" + "=" * 80)
    
    # 解析并可视化metrics.json
    print("\n2. 解析并可视化指标文件 (metrics.json):")
    metrics_info = ModelInfoParserRegistry.parse_file(metrics_path)
    if metrics_info:
        print(f"\n解析成功! 最终测试准确率: {metrics_info.metrics.get('final_test_acc', 'N/A')}")
        ModelInfoVisualizerRegistry.draw(metrics_info)
        # 尝试使用TrainingMetricsVisualizer专门可视化训练指标
        from src.model_visualization.model_info_visualizers import TrainingMetricsVisualizer
        metrics_visualizer = TrainingMetricsVisualizer()
        if metrics_visualizer.support(metrics_info):
            print("\n使用TrainingMetricsVisualizer可视化训练指标:")
            metrics_visualizer.visualize(metrics_info)
    else:
        print("解析失败!")
    
    print("\n" + "=" * 80)
    
    # 解析并可视化pth模型文件
    print("\n3. 解析并可视化模型文件 (.pth):")
    model_info = ModelInfoParserRegistry.parse_file(model_path)
    if model_info:
        print(f"\n解析成功! 模型类型: {model_info.model_type}")
        ModelInfoVisualizerRegistry.draw(model_info)
    else:
        print("解析失败!")
    
    print("\n" + "=" * 80)
    
    # 演示模型比较功能
    print("\n4. 模型比较演示:")
    if config_info and metrics_info and model_info:
        comparison_visualizer = ModelComparisonVisualizer()
        comparison_visualizer.add_model_info(config_info)
        comparison_visualizer.add_model_info(metrics_info)
        comparison_visualizer.add_model_info(model_info)
        
        # 注意：在实际使用中，您应该比较不同的模型，而不是同一个模型的不同表示
        print("\n创建模型比较可视化 (注意：这里仅作为演示，比较的是同一模型的不同表示)")
        comparison_visualizer.visualize(show=True)
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()