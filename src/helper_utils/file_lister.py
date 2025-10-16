import os
import sys
import argparse
import math
import subprocess
from prettytable import PrettyTable

# 打印Python解释器路径和已安装的包
try:
    print(f"调试信息: Python解释器路径: {sys.executable}")
except Exception as e:
    print(f"调试信息: 获取Python信息时出错: {str(e)}")

# 尝试导入视频处理库用于获取视频时长
video_lib_available = False
video_lib_name = ""

# 尝试导入opencv
try:
    import cv2
    # 检查cv2是否有VideoCapture属性
    if hasattr(cv2, 'VideoCapture'):
        video_lib_available = True
        video_lib_name = "opencv"
        print("调试信息: opencv-python已成功导入，VideoCapture可用")
    else:
        print("调试信息: opencv-python已导入，但缺少VideoCapture属性")
        # 尝试检查ffprobe
        raise ImportError("OpenCV缺少VideoCapture功能")
except ImportError:
    print("调试信息: opencv-python未正确安装或缺少必要功能")
    
    # 检查ffprobe是否可用
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        video_lib_available = True
        video_lib_name = "ffprobe"
        print("调试信息: ffprobe已找到，可用于获取视频时长")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("调试信息: ffprobe未找到")
        print("警告: 未找到可用的视频处理库，无法显示视频文件时长")
        print("如需此功能，请安装opencv-python: pip install opencv-python")
        print("或安装ffmpeg(包含ffprobe): https://ffmpeg.org/download.html")

def convert_size(size_bytes):
    """将字节大小转换为人类易读的格式"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def get_file_type(filename):
    """获取文件类型（后缀）"""
    _, ext = os.path.splitext(filename)
    return ext[1:].upper() if ext else "FILE"

def get_video_duration(file_path):
    """获取视频文件的时长，支持opencv和ffprobe两种方式"""
    if not video_lib_available:
        return "无视频库"
    
    # 检查是否为视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if not any(file_path.lower().endswith(ext) for ext in video_extensions):
        return None
    
    try:
        if video_lib_name == "opencv":
            # 再次检查cv2是否有VideoCapture属性，防止运行时错误
            if not hasattr(cv2, 'VideoCapture'):
                print(f"获取视频时长错误 '{file_path}': opencv模块缺少VideoCapture属性")
                return "OpenCV功能缺失"
            
            # 使用opencv获取时长
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return "无法打开"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                minutes, seconds = divmod(int(duration), 60)
                cap.release()
                return f"{minutes:02d}:{seconds:02d}"
            else:
                cap.release()
                return "未知时长"
        
        elif video_lib_name == "ffprobe":
            # 使用ffprobe获取时长
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                minutes, seconds = divmod(int(duration), 60)
                return f"{minutes:02d}:{seconds:02d}"
            else:
                return "获取失败"
                
    except Exception as e:
        # 捕获所有异常，但避免显示详细错误信息，只返回简单提示
        print(f"获取视频时长错误 '{file_path}': {str(e)}")
        return "获取失败"

def list_files(path, extensions=None, recursive=True):
    """
    列出指定路径下的文件
    
    参数:
        path: 目标路径
        extensions: 要筛选的文件后缀列表，None表示所有类型
        recursive: 是否递归遍历子目录
    """
    if not os.path.exists(path):
        print(f"错误: 路径 '{path}' 不存在")
        return
    
    if not os.path.isdir(path):
        print(f"错误: '{path}' 不是一个目录")
        return
    
    # 标准化文件后缀（去除点并转为小写）
    if extensions:
        extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    file_info = {}
    
    # 遍历目录
    for root, dirs, files in os.walk(path):
        file_info[root] = []
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查文件后缀
            if extensions:
                file_ext = os.path.splitext(file)[1].lower().lstrip('.')
                if file_ext not in extensions:
                    continue
            
            # 获取文件信息
            try:
                file_size_bytes = os.path.getsize(file_path)  # 保存原始字节大小
                file_size = convert_size(file_size_bytes)
                file_type = get_file_type(file)
                
                file_data = {
                    'name': file,
                    'type': file_type,
                    'size': file_size,
                    'size_bytes': file_size_bytes  # 新增：保存原始字节数用于汇总
                }
                
                # 尝试获取视频时长
                duration = get_video_duration(file_path)
                if duration:
                    file_data['duration'] = duration
                
                file_info[root].append(file_data)
            except Exception as e:
                print(f"无法获取文件信息 '{file_path}': {e}")
        
        # 如果不递归，只处理当前目录
        if not recursive:
            break
    
    return file_info

def duration_to_seconds(duration_str):
    """将时长字符串(MM:SS)转换为秒数"""
    if not duration_str or ':' not in duration_str:
        return 0
    try:
        minutes, seconds = map(int, duration_str.split(':'))
        return minutes * 60 + seconds
    except ValueError:
        return 0

def seconds_to_duration(total_seconds):
    """将总秒数转换为时长字符串(HH:MM:SS)"""
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def print_file_info(file_info, verbose=False):
    """使用PrettyTable打印文件信息，包含汇总行"""
    for directory, files in file_info.items():
        if files:  # 只打印有符合条件文件的目录
            print(f"\n目录: {directory}")
            print("=" * (len(directory) + 4))
            
            if verbose:
                # 详细模式：显示完整信息和汇总
                table = PrettyTable()
                
                # 检查是否有视频文件
                has_video_files = any('duration' in file for file in files)
                
                # 设置表格列名
                if has_video_files:
                    table.field_names = ["文件名", "类型", "大小", "时长"]
                else:
                    table.field_names = ["文件名", "类型", "大小"]
                
                # 设置表格样式
                table.align["文件名"] = "l"  # 左对齐
                table.align["类型"] = "l"
                table.align["大小"] = "r"  # 右对齐
                if has_video_files:
                    table.align["时长"] = "r"
                
                # 添加文件信息到表格
                total_files = 0
                total_size_bytes = 0
                total_duration_seconds = 0
                video_files_count = 0
                
                for file in files:
                    total_files += 1
                    total_size_bytes += file['size_bytes']
                    
                    # 累加视频时长
                    duration = file.get('duration', '')
                    if duration and ':' in duration:
                        total_duration_seconds += duration_to_seconds(duration)
                        video_files_count += 1
                    
                    if has_video_files:
                        table.add_row([
                            file['name'], 
                            file['type'], 
                            file['size'], 
                            duration
                        ])
                    else:
                        table.add_row([file['name'], file['type'], file['size']])
                
                # 添加分隔线
                table.add_row(["-"*15, "-"*10, "-"*10, "-"*10] if has_video_files else ["-"*15, "-"*10, "-"*10])
                
                # 添加汇总行
                total_size = convert_size(total_size_bytes)
                total_duration = seconds_to_duration(total_duration_seconds)
                
                if has_video_files:
                    if video_files_count > 0:
                        # 有有效的视频文件，显示总时长
                        duration_summary = f"总时长: {total_duration}"
                    else:
                        # 没有有效的视频文件时长
                        duration_summary = ""
                    table.add_row([f"总计: {total_files} 个文件", "", total_size, duration_summary])
                else:
                    table.add_row([f"总计: {total_files} 个文件", "", total_size])
                
                # 打印表格
                print(table)
            else:
                # 摘要模式：只显示文件名和简单汇总
                for file in files:
                    print(f"  - {file['name']}")
                
                # 摘要模式的汇总信息
                total_files = len(files)
                total_size_bytes = sum(file['size_bytes'] for file in files)
                total_size = convert_size(total_size_bytes)
                print(f"\n  该目录下共有 {total_files} 个文件，总大小: {total_size}")
            
            print()  # 添加空行分隔不同目录

def main():
    print("==== 文件列表工具 ====")
    print("该工具可以帮助您列出指定目录下的文件信息")
    print("注意: 显示视频时长需要安装opencv-python或ffmpeg")
    print()
    
    # 交互式获取参数
    while True:
        path = input("请输入目标目录路径: ").strip()
        if path:
            break
        print("错误: 目录路径不能为空，请重新输入")
    
    # 获取文件后缀筛选（可选）
    extensions_input = input("请输入要筛选的文件后缀（多个后缀用空格分隔，留空表示所有类型）: ").strip()
    extensions = extensions_input.split() if extensions_input else None
    
    # 获取是否递归遍历子目录
    recursive_input = input("是否递归遍历子目录？(y/n，默认y): ").strip().lower()
    recursive = recursive_input != 'n'
    
    # 获取是否显示详细信息
    verbose_input = input("是否显示详细信息？(y/n，默认y): ").strip().lower()
    verbose = verbose_input != 'n'  # 默认显示详细信息，除非明确输入n
    
    print("\n正在处理...\n")
    
    # 列出文件并打印信息
    file_info = list_files(
        path=path,
        extensions=extensions,
        recursive=recursive
    )
    
    if file_info:
        print_file_info(file_info, verbose=verbose)
    
    print("\n操作完成！")
    input("按回车键退出...")  # 新增这一行

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")
        input("按回车键退出...")  # 出错后也暂停，让用户看到错误
