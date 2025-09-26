from src.model_visualization.parsers import ModelInfoParserRegistry


def main():
    # 解析配置文件
    config_info = ModelInfoParserRegistry.parse_file(
        file_path="models/model_config.ini",
        namespace="default"
    )
    if config_info:
        print(f"解析配置文件: {config_info.model_name} v{config_info.model_version}")
    
    # 解析指标文件
    metrics_info = ModelInfoParserRegistry.parse_file(
        file_path="models/model_metrics.json",
        namespace="metrics"
    )
    if metrics_info:
        print(f"模型准确率: {metrics_info.accuracy:.2f}, 损失值: {metrics_info.loss:.4f}")
    
    # 解析模型文件
    model_info = ModelInfoParserRegistry.parse_file(
        file_path="models/resnet50.pth",
        namespace="models"
    )
    if model_info:
        print(f"解析模型文件: {model_info.model_name}, 输入形状: {model_info.input_shape}")


if __name__ == "__main__":
    main()
