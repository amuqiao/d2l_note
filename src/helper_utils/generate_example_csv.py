import argparse
import csv
import os
import random
from pathlib import Path

def generate_example_csv(output_filename: str) -> None:
    """
    生成示例CSV文件，包含设备ID、状态和低效原因
    
    Args:
        output_filename: 输出CSV文件的文件名（不含路径时默认在脚本目录生成）
    """
    # 定义可能的数据值
    status_options = ["正常", "异常"]
    reasons = [
        "组串缺失", 
        "阴影遮挡", 
        "频繁跳闸", 
        "灰尘覆盖", 
        "线路老化",
        ""  # 空值表示无故障原因
    ]
    
    # 生成8行数据（含表头）
    data = [
        ["equipment_id", "status", "inefficiency_reason"]  # 表头
    ]
    
    # 生成7行数据
    for i in range(1, 8):
        equipment_id = 100 + i
        status = random.choice(status_options)
        # 正常状态下通常没有故障原因
        reason = random.choice(reasons) if status == "异常" else ""
        
        data.append([equipment_id, status, reason])
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    
    # 如果output_filename是相对路径，则将其放在脚本目录下；如果是绝对路径，则保持不变
    output_path = Path(output_filename)
    if not output_path.is_absolute():
        output_path = script_dir / output_filename
    
    # 写入CSV文件
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        print(f"示例CSV文件已生成在脚本目录：{output_path}")
        print(f"共生成 {len(data)-1} 行数据（不含表头）")
        
    except Exception as e:
        print(f"生成CSV文件失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='在脚本执行目录下生成示例CSV文件，用于测试数据替换脚本')
    parser.add_argument('--output', 
                      default='solar_equipment_data.csv', 
                      help='输出CSV文件名（默认: solar_equipment_data.csv，将生成在脚本执行目录下）')
    
    args = parser.parse_args()
    generate_example_csv(args.output)

if __name__ == "__main__":
    main()
    