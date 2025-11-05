from src.model_show.parser_registry import parse_model_info
import os


def main():
    # 解析配置文件
    # 注意：本示例假设这些文件存在，实际使用时请替换为真实路径
    try:
        # 解析指标文件 - 使用便捷函数
        metrics_file = "runs/run_20250914_044631/metrics.json"
        if os.path.exists(metrics_file):
            metrics_info = parse_model_info(
                file_path=metrics_file,
                namespace="metrics"
            )
            if metrics_info:
                print(f"解析指标文件: {metrics_info.name}")
                print(f"最终测试准确率: {metrics_info.metrics.get('final_test_acc', 0.0):.4f}")
                print(f"最佳测试准确率: {metrics_info.metrics.get('best_test_acc', 0.0):.4f}")
                print(f"最终训练损失: {metrics_info.metrics.get('final_train_loss', 0.0):.6f}")
                print(f"总训练时间: {metrics_info.metrics.get('total_training_time', '0s')}")
        else:
            print(f"示例文件不存在: {metrics_file}")
    except Exception as e:
        print(f"解析过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()
