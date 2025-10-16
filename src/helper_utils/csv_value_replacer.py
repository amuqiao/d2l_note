import argparse
import pandas as pd
import json
from typing import Dict, Any

def load_replacement_rules(rule_str: str) -> Dict[str, Dict[str, Any]]:
    """
    解析替换规则字符串为字典
    
    Args:
        rule_str: 包含替换规则的JSON字符串
        
    Returns:
        解析后的替换规则字典
    """
    try:
        return json.loads(rule_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"替换规则格式错误: {str(e)}")

def replace_values_in_csv(input_file: str, output_file: str, replacement_rules: Dict[str, Dict[str, Any]]) -> None:
    """
    读取CSV文件，根据替换规则替换值，并保存到新文件
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        replacement_rules: 替换规则字典，格式为{列名: {原始值: 目标值}}
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {str(e)}")
    
    # 检查替换规则中的列是否存在于CSV中
    for column in replacement_rules:
        if column not in df.columns:
            raise ValueError(f"CSV文件中不存在列: {column}")
    
    # 执行替换操作
    for column, value_map in replacement_rules.items():
        df[column] = df[column].replace(value_map)
    
    # 保存修改后的CSV文件
    try:
        df.to_csv(output_file, index=False)
        print(f"成功将修改后的数据保存到: {output_file}")
    except Exception as e:
        raise RuntimeError(f"保存CSV文件失败: {str(e)}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='读取CSV文件并根据指定规则替换列值',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--rules', required=True, 
                        help='替换规则，JSON格式字符串。例如:\n'
                             '{\n'
                             '  "inefficiency_reason": {\n'
                             '    "组串缺失": "xx",\n'
                             '    "阴影遮挡": "xxx",\n'
                             '    "频繁跳闸": "xxxx"\n'
                             '  },\n'
                             '  "status": {\n'
                             '    "正常": "normal",\n'
                             '    "异常": "abnormal"\n'
                             '  }\n'
                             '}\n\n示例命令:\n'
                             'python csv_value_replacer.py --input solar_equipment_data.csv --output solar_equipment_data.csv --rules \'{"inefficiency_reason": {"线路老化": "线路优化", "组串缺失": "缺失"}}\'')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 加载替换规则
        rules = load_replacement_rules(args.rules)
        
        # 执行替换操作
        replace_values_in_csv(args.input, args.output, rules)
        
        print("数据替换完成！")
    except Exception as e:
        print(f"操作失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
