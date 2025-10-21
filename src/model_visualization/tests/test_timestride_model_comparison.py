import sys
import os
import argparse
import glob
import fnmatch
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 确保这行正确添加了项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加当前目录

from src.model_visualization.model_info_visualizers_impl.timestride_model_comparison_visualizer import TimestrideModelComparisonVisualizer


def parse_arguments():
    """
    🧩 解析命令行参数
    
    返回:
        argparse.Namespace: 解析后的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="📊 Timestride模型比较可视化工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
  # 使用默认timestride运行目录
  python test_timestride_model_comparison.py
  
  # 指定自定义运行目录
  python test_timestride_model_comparison.py --model-dir "/data/home/project/d2l_note/timestride/cp_0924_count"
  
  # 递归查找模型目录
  python test_timestride_model_comparison.py --model-dir "/data/home/project/d2l_note/timestride" --recursive
  
  # 启用详细输出模式
  python test_timestride_model_comparison.py --model-dir "/data/home/project/d2l_note/timestride" --recursive -v
  
  # 设置模型比较的排序方式
  python test_timestride_model_comparison.py --model-dir "/data/home/project/d2l_note/timestride" --recursive --sort-by "name"
  
  # 显示帮助信息
  python test_timestride_model_comparison.py -h
        """)
    
    # 修改参数名称为更通用的model-dir
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="/data/home/project/d2l_note/timestride/cp_0924_count",
        help="Timestride模型目录路径，默认为 '/data/home/project/d2l_note/timestride/cp_0924_count'"
    )
    
    # 添加递归查找选项
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归查找模型目录"
    )
    
    # 添加详细输出选项
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="启用详细输出模式"
    )
    
    # 添加排序方式选项
    parser.add_argument(
        "--sort-by",
        type=str,
        default="accuracy",
        choices=["accuracy", "name"],
        help="模型比较的排序方式，默认为 'accuracy'"
    )
    
    # 添加可视化类型选项
    parser.add_argument(
        "--plot-type",
        type=str,
        default="ranking",
        choices=["ranking"],
        help="可视化类型，默认为 'ranking'，目前仅支持ranking模式"
    )
    
    return parser.parse_args()

def find_model_directories(root_dir, recursive=False):
    """
    🔍 查找包含timestride模型文件的目录
    
    参数:
        root_dir (str): 根目录路径
        recursive (bool): 是否递归查找
        
    返回:
        list: 包含模型文件的目录列表
    """
    model_dirs = []
    
    if recursive:
        # 递归查找包含args.json文件的目录
        for dirpath, _, filenames in os.walk(root_dir):
            # 检查是否包含args.json文件（这是timestride模型的特征）
            has_args = any(fnmatch.fnmatch(f, 'args.json') for f in filenames)
            has_training_metrics = any(fnmatch.fnmatch(f, 'training_metrics.json') for f in filenames)
            
            # 检查是否有test_results目录
            has_test_results = os.path.exists(os.path.join(dirpath, 'test_results'))
            
            if has_args or has_training_metrics or has_test_results:
                model_dirs.append(dirpath)
    else:
        # 仅检查指定目录下的子目录
        if os.path.isdir(root_dir):
            subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            for subdir in subdirs:
                # 检查子目录中是否有args.json文件
                has_args = os.path.exists(os.path.join(subdir, 'args.json'))
                has_training_metrics = os.path.exists(os.path.join(subdir, 'training_metrics.json'))
                has_test_results = os.path.exists(os.path.join(subdir, 'test_results'))
                
                if has_args or has_training_metrics or has_test_results:
                    model_dirs.append(subdir)
    
    return model_dirs

def collect_model_info(model_dir, verbose, visualizer):
    """
    📁 收集单个模型目录的信息，用于后续比较
    
    参数:
        model_dir (str): 模型目录路径
        verbose (bool): 是否启用详细输出
        visualizer: TimestrideModelComparisonVisualizer实例
        
    返回:
        ModelInfoData: 解析后的模型信息对象，如果解析失败则返回None
    """
    print(f"\n🔍 处理模型目录: {model_dir}")
    
    try:
        # 使用visualizer的方法从路径创建模型信息
        model_info = visualizer.create_model_info_from_path(model_dir)
        
        if model_info:
            # 检查是否包含准确率信息
            if "accuracy" in model_info.metrics:
                print(f"✅ 解析成功! 模型准确率: {model_info.metrics['accuracy']:.4f}")
                return model_info
            else:
                print(f"⚠️ 警告：该模型信息不包含准确率指标")
                if verbose:
                    print(f"[详细信息] 模型指标: {model_info.metrics}")
        else:
            print(f"❌ 解析失败: 无法从路径创建模型信息")
    except Exception as e:
        print(f"❌ 解析模型目录出错: {str(e)}")
    
    return None

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
    🚀 主函数：协调Timestride模型比较可视化过程
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印程序标题
    print("=" * 80)
    print("📈 Timestride模型比较可视化工具")
    print("=" * 80)
    print_verbose(args.verbose, f"使用的模型目录: {args.model_dir}")
    print_verbose(args.verbose, f"递归查找: {'启用' if args.recursive else '禁用'}")
    print_verbose(args.verbose, f"模型比较排序方式: {'按准确率' if args.sort_by == 'accuracy' else '按名称'}")
    
    # 查找模型目录
    model_dirs = find_model_directories(args.model_dir, args.recursive)
    
    if not model_dirs:
        print(f"❌ 错误：在指定路径 {args.model_dir} 中未找到任何Timestride模型目录")
        return
    
    print(f"✅ 找到 {len(model_dirs)} 个模型目录")
    for i, model_dir in enumerate(model_dirs, 1):
        print_verbose(args.verbose, f"{i}. {model_dir}")
    
    print("\n" + "=" * 80)
    
    # 使用Timestride模型比较可视化器
    print("🆚 使用TimestrideModelComparisonVisualizer对比多个模型")
    comparison_visualizer = TimestrideModelComparisonVisualizer()
    
    # 设置排序方式
    comparison_visualizer.set_sort_by(args.sort_by)
    
    # 为每个目录收集模型信息
    valid_model_count = 0
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"\n🔍 处理模型目录 {i}/{len(model_dirs)}: {model_dir}")
        model_info = collect_model_info(model_dir, args.verbose, comparison_visualizer)
        
        if model_info:
            comparison_visualizer.add_model_info(model_info)
            valid_model_count += 1
            print_verbose(args.verbose, f"已添加有效的模型信息 #{valid_model_count}")
    
    # 执行模型比较可视化
    try:
        print("\n" + "=" * 80)
        print("📊 Timestride多个模型比较结果")
        print(f"🔍 使用{args.plot_type}模式: 显示详细准确率比较，包括模型类型、参数和来源路径")
        
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


if __name__ == "__main__":
    main()