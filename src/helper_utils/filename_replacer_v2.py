import os
import sys

# 处理打包成exe后的路径问题
if getattr(sys, 'frozen', False):
    # 运行在打包后的exe中
    CURRENT_DIR = os.path.dirname(sys.executable)
else:
    # 运行在普通Python环境中
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 添加一个获取正确路径的辅助函数
def get_correct_path(path):
    """确保获取正确的文件或目录路径"""
    # 如果是相对路径，基于当前工作目录解析
    if not os.path.isabs(path):
        if getattr(sys, 'frozen', False):
            # 在exe模式下，使用当前工作目录而不是exe所在目录
            return os.path.abspath(os.path.join(os.getcwd(), path))
        else:
            return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    return path

def rename_files(directory, target, replacement, recursive=False, dry_run=False):
    """
    在指定目录下重命名文件，将文件名中的目标内容替换为指定内容
    
    Args:
        directory (str): 目录路径
        target (str): 需要替换的目标内容
        replacement (str): 替换内容
        recursive (bool): 是否递归处理子目录
        dry_run (bool): 仅显示操作，不实际执行
    """
    # 获取正确的目录路径
    corrected_directory = get_correct_path(directory)
    
    # 检查目录是否存在
    if not os.path.isdir(corrected_directory):
        print(f"错误：目录 '{directory}' 不存在", file=sys.stderr)
        return
    
    # 使用修正后的路径
    directory = corrected_directory
    
    # 遍历目录中的所有条目
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        
        # 如果是目录且需要递归处理，则递归调用
        if os.path.isdir(entry_path) and recursive:
            rename_files(entry_path, target, replacement, recursive, dry_run)
        # 如果是文件，则处理文件名
        elif os.path.isfile(entry_path):
            # 检查文件名是否包含目标内容
            if target in entry:
                # 生成新的文件名
                new_entry = entry.replace(target, replacement)
                new_entry_path = os.path.join(directory, new_entry)
                
                # 显示或执行重命名操作
                if dry_run:
                    print(f"将要重命名: {entry} -> {new_entry}")
                else:
                    try:
                        os.rename(entry_path, new_entry_path)
                        print(f"已重命名: {entry} -> {new_entry}")
                    except Exception as e:
                        print(f"重命名失败 '{entry}': {str(e)}", file=sys.stderr)

def main():
    print("==== 文件名替换工具 ====")
    print("该工具可以帮助您替换指定目录下文件名中的特定内容")
    print()
    
    # 交互式获取参数
    while True:
        directory = input("请输入目标目录路径: ").strip()
        if directory:
            break
        print("错误: 目录路径不能为空，请重新输入")
    
    # 获取要替换的内容
    while True:
        target = input("请输入需要替换的内容: ").strip()
        if target:
            break
        print("错误: 需要替换的内容不能为空，请重新输入")
    
    # 获取替换内容（可以为空）
    replacement = input("请输入替换内容（留空表示删除）: ").strip()
    
    # 获取是否递归遍历子目录
    recursive_input = input("是否递归遍历子目录？(y/n，默认n): ").strip().lower()
    recursive = recursive_input == 'y'
    
    # 获取是否执行模拟运行
    dry_run_input = input("是否执行模拟运行（仅显示将要执行的操作，不实际修改文件）？(y/n，默认n): ").strip().lower()
    dry_run = dry_run_input == 'y'
    
    print("\n正在处理...\n")
    
    # 显示操作信息
    print(f"在目录 '{directory}' 中")
    print(f"将文件名中的 '{target}' 替换为 '{replacement}'")
    if recursive:
        print("将递归处理所有子目录")
    if dry_run:
        print("这是一个模拟运行，不会实际修改文件")
    
    # 确认操作
    if not dry_run:
        confirm = input("是否继续? (y/n) ")
        if confirm.lower() not in ['y', 'yes']:
            print("操作已取消")
            return
    
    # 处理目录路径，确保在exe环境中也能正确工作
    directory = get_correct_path(directory)
    
    # 调用重命名函数进行模拟运行或实际运行
    rename_files(directory, target, replacement, recursive, dry_run)
    
    # 如果是模拟运行，询问是否执行实际重命名
    if dry_run:
        print("\n模拟运行完成！")
        confirm_actual = input("是否执行实际的重命名操作？(y/n) ")
        if confirm_actual.lower() in ['y', 'yes']:
            print("\n正在执行实际重命名...\n")
            rename_files(directory, target, replacement, recursive, False)
    
    print("\n操作完成！")
    input("按回车键退出...")  # 添加暂停，确保用户可以看到输出

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        input("按回车键退出...")  # 出错后也暂停，让用户看到错误
