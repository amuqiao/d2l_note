import sys
import os
import argparse
import glob
import fnmatch
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 确保这行正确添加了项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加当前目录

from src.model_visualization.model_info_parsers import ModelInfoParserRegistry, register_parsers
from src.model_visualization.model_info_visualizers import ModelInfoVisualizerRegistry, register_visualizers
from src.model_visualization.model_info_visualizers_impl import (
    ModelComparisonVisualizer,
    TrainingMetricsVisualizer,
    ModelSummaryVisualizer
)

# 注册解析器
register_parsers()
# 注册可视化器
register_visualizers()


def parse_arguments():
    """
    🧩 解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="📊 模型信息解析与可视化工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
  # 使用默认运行目录和命名空间
  python test_model_info_v1.py
  
  # 指定自定义运行目录
  python test_model_info_v1.py --model-dir "runs2/run_20250909_204054"
  
  # 递归查找模型目录
  python test_model_info_v1.py --model-dir "runs2" --recursive
  
  # 指定命名空间
  python test_model_info_v1.py --namespace "project1"
  
  # 启用详细输出模式
  python test_model_info_v1.py -v
  
  # 设置模型比较的图表类型
  python test_model_info_v1.py --model-dir "runs2" --recursive --plot-type "ranking"
  
  # 设置模型比较的排序方式
  python test_model_info_v1.py --model-dir "runs2" --recursive --sort-by "name"
  
  # 使用增强版ranking模式比较模型准确率
  python test_model_info_v1.py --model-dir "runs2" --recursive --plot-type "ranking" --sort-by "accuracy"
  
  # 显示帮助信息
  python test_model_info_v1.py -h
        """)
    
    # 修改参数名称为更通用的model-dir
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="runs/run_20250914_040635",
        help="模型目录路径，默认为 'runs/run_20250914_040635'"
    )
    
    # 添加递归查找选项
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归查找模型目录"
    )
    
    # 添加命名空间参数
    parser.add_argument(
        "--namespace", 
        type=str, 
        default="default",
        help="模型命名空间，默认为 'default'"
    )
    
    # 添加详细输出选项
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="启用详细输出模式"
    )
    
    # 添加图表类型选项
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "bar", "ranking", "scatter"],
        help="模型比较的图表类型，默认为 'all'。ranking模式已优化为详细比较模型准确率"
    )
    
    # 添加排序方式选项
    parser.add_argument(
        "--sort-by",
        type=str,
        default="accuracy",
        choices=["accuracy", "name"],
        help="模型比较的排序方式，默认为 'accuracy'"
    )
    
    return parser.parse_args()

def find_model_directories(root_dir, recursive=False):
    """
    🔍 查找包含模型文件的目录
    
    参数:
        root_dir (str): 根目录路径
        recursive (bool): 是否递归查找
        
    返回:
        list: 包含模型文件的目录列表
    """
    model_dirs = []
    
    if recursive:
        # 递归查找包含配置文件或模型文件的目录
        for dirpath, _, filenames in os.walk(root_dir):
            # 检查是否包含配置文件或模型文件
            has_config = any(fnmatch.fnmatch(f, 'config.json') for f in filenames)
            has_metrics = any(fnmatch.fnmatch(f, 'metrics.json') for f in filenames)
            has_model = any(fnmatch.fnmatch(f, '*.pth') for f in filenames)
            
            if has_config or has_metrics or has_model:
                model_dirs.append(dirpath)
    else:
        # 仅检查指定目录
        if os.path.isdir(root_dir):
            filenames = os.listdir(root_dir)
            has_config = any(fnmatch.fnmatch(f, 'config.json') for f in filenames)
            has_metrics = any(fnmatch.fnmatch(f, 'metrics.json') for f in filenames)
            has_model = any(fnmatch.fnmatch(f, '*.pth') for f in filenames)
            
            if has_config or has_metrics or has_model:
                model_dirs.append(root_dir)
    
    return model_dirs


def process_single_model_directory(model_dir, namespace, verbose):
    """
    📁 处理单个模型目录，使用基础可视化器
    
    参数:
        model_dir (str): 模型目录路径
        namespace (str): 命名空间
        verbose (bool): 是否启用详细输出
    """
    print(f"\n🔍 处理模型目录: {model_dir}")
    
    # 构建文件路径
    config_path = os.path.join(model_dir, "config.json")
    metrics_path = os.path.join(model_dir, "metrics.json")
    
    # 动态查找.pth结尾的模型文件
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if model_files:
        # 优先选择包含'best'的模型文件
        best_model_files = [f for f in model_files if 'best' in os.path.basename(f).lower()]
        if best_model_files:
            model_path = best_model_files[0]  # 选择第一个找到的最佳模型
        else:
            model_path = model_files[0]  # 如果没有找到'best'模型，选择第一个.pth文件
    else:
        model_path = None
        print(f"⚠️ 警告：在目录 {model_dir} 中未找到.pth结尾的模型文件")
    
    print_verbose(verbose, f"配置文件路径: {config_path}")
    print_verbose(verbose, f"指标文件路径: {metrics_path}")
    print_verbose(verbose, f"模型文件路径: {model_path}")
    
    # 存储解析的信息
    parsed_infos = []
    
    # 使用ModelSummaryVisualizer作为基础可视化器
    summary_visualizer = ModelSummaryVisualizer()
    
    # 解析并可视化配置文件
    try:
        if os.path.exists(config_path):
            config_info = ModelInfoParserRegistry.parse_file(config_path, namespace=namespace)
            if config_info:
                print(f"\n✅ 解析成功! 配置文件: {os.path.basename(config_path)}")
                summary_visualizer.visualize(config_info)
                parsed_infos.append(config_info)
                print_verbose(verbose, "配置文件解析完成并成功可视化")
            else:
                print(f"❌ 解析失败: {os.path.basename(config_path)}")
    except Exception as e:
        print(f"❌ 解析配置文件出错: {str(e)}")
    
    # 解析并可视化指标文件
    try:
        if os.path.exists(metrics_path):
            metrics_info = ModelInfoParserRegistry.parse_file(metrics_path, namespace=namespace)
            if metrics_info:
                final_acc = metrics_info.metrics.get('final_test_acc', 'N/A')
                print(f"\n✅ 解析成功! 指标文件: {os.path.basename(metrics_path)}, 最终测试准确率: {final_acc}")
                summary_visualizer.visualize(metrics_info)
                
                # 尝试使用专门的训练指标可视化器
                metrics_visualizer = TrainingMetricsVisualizer()
                if metrics_visualizer.support(metrics_info):
                    print("\n📈 使用TrainingMetricsVisualizer可视化训练指标:")
                    metrics_visualizer.visualize(metrics_info)
                    print_verbose(verbose, "训练指标已使用专门的可视化器展示")
                
                parsed_infos.append(metrics_info)
            else:
                print(f"❌ 解析失败: {os.path.basename(metrics_path)}")
    except Exception as e:
        print(f"❌ 解析指标文件出错: {str(e)}")
    
    # 解析并可视化模型文件
    try:
        if model_path and os.path.exists(model_path):
            model_info = ModelInfoParserRegistry.parse_file(model_path, namespace=namespace)
            if model_info:
                print(f"\n✅ 解析成功! 模型文件: {os.path.basename(model_path)}")
                summary_visualizer.visualize(model_info)
                parsed_infos.append(model_info)
                print_verbose(verbose, "模型文件解析完成并成功可视化")
            else:
                print(f"❌ 解析失败: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"❌ 解析模型文件出错: {str(e)}")
    
    return parsed_infos


def print_verbose(verbose, message):
    """
    💬 条件性打印详细信息
    
    参数:
        verbose (bool): 是否启用详细输出
        message (str): 要打印的消息
    """
    if verbose:
        print(f"[详细信息] {message}")


def main():
    """
    🚀 主函数：协调模型信息的解析与可视化过程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印程序标题
    print("=" * 80)
    print("📈 模型信息解析器和可视化器")
    print("=" * 80)
    print_verbose(args.verbose, f"使用的模型目录: {args.model_dir}")
    print_verbose(args.verbose, f"递归查找: {'启用' if args.recursive else '禁用'}")
    print_verbose(args.verbose, f"使用的命名空间: {args.namespace}")
    
    # 查找模型目录
    model_dirs = find_model_directories(args.model_dir, args.recursive)
    
    if not model_dirs:
        print(f"❌ 错误：在指定路径 {args.model_dir} 中未找到任何模型目录")
        return
    
    print(f"✅ 找到 {len(model_dirs)} 个模型目录")
    for i, model_dir in enumerate(model_dirs, 1):
        print_verbose(args.verbose, f"{i}. {model_dir}")
    
    print("\n" + "=" * 80)
    
    # 根据找到的模型目录数量决定使用哪种可视化器
    if len(model_dirs) == 1:
        # 单个模型目录，使用基础可视化器
        print("📋 使用基础可视化器展示单个模型信息")
        process_single_model_directory(model_dirs[0], args.namespace, args.verbose)
    else:
        # 多个模型目录，使用比较可视化器
        print("🆚 使用比较可视化器对比多个模型")
        comparison_visualizer = ModelComparisonVisualizer()
        
        # 设置排序方式
        comparison_visualizer.set_sort_by(args.sort_by)
        print_verbose(args.verbose, f"模型比较排序方式: {'按准确率' if args.sort_by == 'accuracy' else '按名称'}")
        print_verbose(args.verbose, f"模型比较图表类型: {args.plot_type}")
        
        # 为每个目录创建一个综合的模型信息对象
        valid_model_count = 0
        for i, model_dir in enumerate(model_dirs, 1):
            print(f"\n🔍 处理模型目录 {i}/{len(model_dirs)}: {model_dir}")
            parsed_infos = process_single_model_directory(model_dir, f"{args.namespace}_{i}", args.verbose)
            
            # 优先选择包含测试准确率的信息作为比较基础
            if parsed_infos:
                # 查找包含测试准确率的指标信息
                metrics_info = next((info for info in parsed_infos if "final_test_acc" in info.metrics or "best_acc" in info.metrics), None)
                if metrics_info:
                    comparison_visualizer.add_model_info(metrics_info)
                    valid_model_count += 1
                    print_verbose(args.verbose, f"已添加有效的模型信息 #{valid_model_count} (包含准确率指标)")
                else:
                    # 如果没有指标信息，尝试使用第一个可用的信息
                    print_verbose(args.verbose, "警告: 该模型信息不包含准确率指标，可能会影响比较结果")
                    comparison_visualizer.add_model_info(parsed_infos[0])
                    valid_model_count += 1
        
        # 执行模型比较可视化
        try:
            print("\n" + "=" * 80)
            print("📊 多个模型比较结果")
            
            # 特别说明ranking模式的增强功能
            if args.plot_type == "ranking":
                print("🔍 增强版ranking模式: 显示详细准确率比较，包括测试/训练/验证准确率及差异分析")
            
            # 执行可视化
            result = comparison_visualizer.visualize(show=True, plot_type=args.plot_type)
            
            if result:
                print_verbose(args.verbose, "模型比较可视化完成")
            else:
                print("⚠️ 警告：未能生成有效的比较")
        except Exception as e:
            print(f"❌ 模型比较过程出错: {str(e)}")
            print_verbose(args.verbose, f"模型比较错误详情: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ 处理完成!")
    print("=" * 80)

def namespace_demo():
    """
    🌌 命名空间功能演示
    
    本函数演示如何在同一程序中使用不同命名空间的模型
    """
    print("=" * 80)
    print("🌌 命名空间功能演示")
    print("=" * 80)
    
    # 示例：使用两个不同命名空间解析同一文件
    # 注意：在实际应用中，通常会为不同项目的模型使用不同的命名空间
    demo_file_path = "runs/run_20250914_040635/config.json"  # 假设这个路径存在
    
    if os.path.exists(demo_file_path):
        # 使用默认命名空间解析
        default_namespace_info = ModelInfoParserRegistry.parse_file(demo_file_path, namespace="default")
        print(f"\n📁 使用 'default' 命名空间解析文件:")
        if default_namespace_info:
            print(f"   - 类型: {default_namespace_info.type}")
            print(f"   - 路径: {default_namespace_info.path}")
            print(f"   - 命名空间: {default_namespace_info.namespace}")
            print(f"   - 模型类型: {default_namespace_info.model_type}")
        
        # 使用自定义命名空间解析
        custom_namespace_info = ModelInfoParserRegistry.parse_file(demo_file_path, namespace="projectA")
        print(f"\n📁 使用 'projectA' 命名空间解析文件:")
        if custom_namespace_info:
            print(f"   - 类型: {custom_namespace_info.type}")
            print(f"   - 路径: {custom_namespace_info.path}")
            print(f"   - 命名空间: {custom_namespace_info.namespace}")
            print(f"   - 模型类型: {custom_namespace_info.model_type}")
        
        # 可视化不同命名空间的模型信息
        print(f"\n� 可视化不同命名空间的模型信息:")
        if default_namespace_info:
            ModelInfoVisualizerRegistry.draw(default_namespace_info)
        if custom_namespace_info:
            ModelInfoVisualizerRegistry.draw(custom_namespace_info)
        
        print("\n�💡 提示: 在实际应用中，您可以使用命名空间来区分不同项目的模型，")
        print("   即使模型文件名称或路径相似，也可以通过命名空间进行有效隔离。")
    else:
        print(f"⚠️  示例文件不存在: {demo_file_path}")
        print("   请确保指定的运行目录包含配置文件。")
    
    print("\n" + "=" * 80)
    print("✅ 命名空间演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
    # 如果需要单独运行命名空间演示，可以取消下面一行的注释
    # namespace_demo()
