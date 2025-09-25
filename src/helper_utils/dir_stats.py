import argparse
import os
from collections import defaultdict

def get_file_extension(filename):
    """获取文件的扩展名（不含点），无扩展名则返回'no_extension'"""
    ext = os.path.splitext(filename)[1]
    return ext[1:].lower() if ext else 'no_extension'

def traverse_directory(root_dir):
    """
    递归遍历目录，统计文件和目录信息
    
    返回值:
    - total_files: 总文件数
    - total_dirs: 总目录数
    - root_info: 根目录信息
    - first_level_dirs: 一级目录信息列表
    - second_level_dirs: 二级目录信息列表
    - all_file_types: 所有文件的类型统计
    """
    # 初始化统计变量
    total_files = 0
    total_dirs = 0
    first_level_dirs = []
    second_level_dirs = []
    all_file_types = defaultdict(int)
    
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"警告: 目录 '{root_dir}' 不存在")
        return (0, 0, None, [], [], {})
    
    if not os.path.isdir(root_dir):
        print(f"警告: '{root_dir}' 不是一个目录")
        return (0, 0, None, [], [], {})
    
    try:
        # 获取根目录下的所有条目
        entries = os.listdir(root_dir)
    except PermissionError:
        print(f"警告: 没有访问 '{root_dir}' 的权限")
        return (0, 0, None, [], [], {})
    except Exception as e:
        print(f"警告: 访问 '{root_dir}' 时发生错误: {str(e)}")
        return (0, 0, None, [], [], {})
    
    # 统计根目录下的文件
    root_files = 0
    root_file_types = defaultdict(int)
    
    for entry in entries:
        entry_path = os.path.join(root_dir, entry)
        
        try:
            if os.path.isfile(entry_path):
                # 统计根目录下的文件
                root_files += 1
                ext = get_file_extension(entry)
                root_file_types[ext] += 1
                all_file_types[ext] += 1
            elif os.path.isdir(entry_path):
                # 处理一级目录
                total_dirs += 1
                
                # 统计一级目录信息
                first_level_info = {
                    'name': entry,
                    'path': entry_path,
                    'files': 0,          # 本级文件数
                    'subdirs': 0,        # 二级目录数
                    'total_files': 0,    # 本级+子目录总文件数
                    'file_types': defaultdict(int)
                }
                
                try:
                    sub_entries = os.listdir(entry_path)
                except PermissionError:
                    print(f"警告: 没有访问一级目录 '{entry_path}' 的权限")
                    first_level_dirs.append(first_level_info)
                    continue
                except Exception as e:
                    print(f"警告: 访问一级目录 '{entry_path}' 时发生错误: {str(e)}")
                    first_level_dirs.append(first_level_info)
                    continue
                
                for sub_entry in sub_entries:
                    sub_entry_path = os.path.join(entry_path, sub_entry)
                    
                    try:
                        if os.path.isfile(sub_entry_path):
                            # 统计一级目录下的文件
                            first_level_info['files'] += 1
                            first_level_info['total_files'] += 1
                            ext = get_file_extension(sub_entry)
                            first_level_info['file_types'][ext] += 1
                            all_file_types[ext] += 1
                        elif os.path.isdir(sub_entry_path):
                            # 处理二级目录
                            first_level_info['subdirs'] += 1
                            total_dirs += 1
                            
                            # 统计二级目录信息
                            second_level_info = {
                                'parent': entry,
                                'name': sub_entry,
                                'path': sub_entry_path,
                                'files': 0,
                                'file_types': defaultdict(int)
                            }
                            
                            try:
                                sub_sub_entries = os.listdir(sub_entry_path)
                            except PermissionError:
                                print(f"警告: 没有访问二级目录 '{sub_entry_path}' 的权限")
                                second_level_dirs.append(second_level_info)
                                continue
                            except Exception as e:
                                print(f"警告: 访问二级目录 '{sub_entry_path}' 时发生错误: {str(e)}")
                                second_level_dirs.append(second_level_info)
                                continue
                            
                            for sub_sub_entry in sub_sub_entries:
                                sub_sub_entry_path = os.path.join(sub_entry_path, sub_sub_entry)
                                
                                try:
                                    if os.path.isfile(sub_sub_entry_path):
                                        # 统计二级目录下的文件
                                        second_level_info['files'] += 1
                                        first_level_info['total_files'] += 1
                                        ext = get_file_extension(sub_sub_entry)
                                        second_level_info['file_types'][ext] += 1
                                        first_level_info['file_types'][ext] += 1
                                        all_file_types[ext] += 1
                                except Exception as e:
                                    print(f"警告: 访问文件 '{sub_sub_entry_path}' 时发生错误: {str(e)}")
                            
                            second_level_dirs.append(second_level_info)
                    except Exception as e:
                        print(f"警告: 访问条目 '{sub_entry_path}' 时发生错误: {str(e)}")
                
                first_level_dirs.append(first_level_info)
        except Exception as e:
            print(f"警告: 访问条目 '{entry_path}' 时发生错误: {str(e)}")
    
    # 计算总文件数
    total_files = root_files + sum(dir_info['total_files'] for dir_info in first_level_dirs)
    
    # 根目录本身也是一个目录
    total_dirs += 1  # 加上根目录
    
    # 根目录信息
    root_info = {
        'name': os.path.basename(root_dir),
        'path': root_dir,
        'files': root_files,
        'subdirs': len(first_level_dirs),
        'total_files': total_files,
        'file_types': root_file_types
    }
    
    return (total_files, total_dirs, root_info, first_level_dirs, second_level_dirs, all_file_types)

def format_file_types(file_types, total, show_percent=True):
    """格式化文件类型统计信息，按数量降序排列"""
    if not file_types:
        return "无文件"
    
    items = []
    # 按文件数量降序排列
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        if show_percent and total > 0:
            percent = (count / total) * 100
            items.append(f"{ext}: {count} ({percent:.1f}%)")
        else:
            items.append(f"{ext}: {count}")
    
    return ", ".join(items)

def calculate_width(text):
    """计算文本的显示宽度，考虑中文字符占2个宽度"""
    width = 0
    for char in text:
        # 中文字符的Unicode范围
        if '\u4e00' <= char <= '\u9fff':
            width += 2
        else:
            width += 1
    return width

def format_column(text, width):
    """根据显示宽度格式化文本"""
    text_width = calculate_width(str(text))
    padding = max(0, width - text_width)
    return str(text) + ' ' * padding

def print_statistics(total_files, total_dirs, root_info, first_level_dirs, second_level_dirs, all_file_types):
    """以清晰易读的格式打印统计结果"""
    if not root_info:
        return
    
    # 计算一级和二级目录数量
    first_level_count = len(first_level_dirs)
    second_level_count = len(second_level_dirs)
    
    print("=" * 80)
    print(f"目录统计结果: {root_info['path']}")
    print("=" * 80)
    print(f"总文件数: {total_files}")
    print(f"总目录数: {total_dirs} (一级目录: {first_level_count}, 二级目录: {second_level_count})")
    print()
    
    # 打印整体文件类型分布
    print("整体文件类型分布:")
    if total_files > 0:
        print(f"  {format_file_types(all_file_types, total_files)}")
    else:
        print("  无文件")
    print()
    
    # 打印一级目录详细信息
    print("一级目录详细信息:")
    
    # 定义各列的宽度，考虑中文标题的宽度
    col1_width = 20  # 目录名
    col2_width = 12  # 本级文件数
    col3_width = 12  # 二级目录数
    col4_width = 12  # 总文件数
    
    # 打印表头
    header = (f"  {format_column('目录名', col1_width)}" +
              f"{format_column('本级文件数', col2_width)}" +
              f"{format_column('二级目录数', col3_width)}" +
              f"{format_column('总文件数', col4_width)}" +
              "文件类型分布")
    print(header)
    print("  " + "-" * 76)
    
    # 先打印根目录
    root_file_types = format_file_types(root_info['file_types'], root_info['total_files'], show_percent=False)
    root_line = (f"  {format_column('[根目录] ' + root_info['name'], col1_width)}" +
                 f"{format_column(root_info['files'], col2_width)}" +
                 f"{format_column(root_info['subdirs'], col3_width)}" +
                 f"{format_column(root_info['total_files'], col4_width)}" +
                 root_file_types)
    print(root_line)
    
    # 再打印其他一级目录（按总文件数从大到小排序）
    for dir_info in sorted(first_level_dirs, key=lambda x: x['total_files'], reverse=True):
        file_types_str = format_file_types(dir_info['file_types'], dir_info['total_files'], show_percent=False)
        dir_line = (f"  {format_column(dir_info['name'], col1_width)}" +
                    f"{format_column(dir_info['files'], col2_width)}" +
                    f"{format_column(dir_info['subdirs'], col3_width)}" +
                    f"{format_column(dir_info['total_files'], col4_width)}" +
                    file_types_str)
        print(dir_line)
    
    print()
    
    # 打印二级目录详细信息
    print("二级目录详细信息:")
    
    # 定义二级目录表格各列的宽度
    col1_width = 30  # 父目录/目录名
    col2_width = 12  # 文件数
    
    # 打印二级目录表头
    header2 = (f"  {format_column('父目录/目录名', col1_width)}" +
               f"{format_column('文件数', col2_width)}" +
               "文件类型分布及百分比")
    print(header2)
    print("  " + "-" * 76)
    
    # 按文件数从大到小排序
    for dir_info in sorted(second_level_dirs, key=lambda x: x['files'], reverse=True):
        dir_name = f"{dir_info['parent']}/{dir_info['name']}"
        file_types_str = format_file_types(dir_info['file_types'], dir_info['files'])
        dir_line2 = (f"  {format_column(dir_name, col1_width)}" +
                    f"{format_column(dir_info['files'], col2_width)}" +
                    file_types_str)
        print(dir_line2)
    
    print("\n" + "=" * 80)
    
def parse_args():
    """解析命令行参数，获取根目录路径"""
    parser = argparse.ArgumentParser(description='统计目录结构和文件类型分布信息')
    parser.add_argument('--root_dir', default='/data/home/project/d2l_note', help='要分析的根目录路径')
    return parser.parse_args()

def main():
    args = parse_args()
    root_dir = args.root_dir
    
    # 遍历目录并获取统计信息
    stats = traverse_directory(root_dir)
    total_files, total_dirs, root_info, first_level_dirs, second_level_dirs, all_file_types = stats
    
    # 打印统计结果
    print_statistics(total_files, total_dirs, root_info, first_level_dirs, second_level_dirs, all_file_types)

if __name__ == "__main__":
    main()
