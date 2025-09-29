from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from typing import List, Optional
from src.model_visualization.data_models import ModelInfoData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils.log_utils import get_logger
from prettytable import PrettyTable

# 导入基类
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class ModelComparisonVisualizer(BaseModelInfoVisualizer):
    """模型准确率比较可视化器：专注于比较多个模型的准确率指标"""
    
    def __init__(self):
        self.model_infos = []
        # 新增：设置排序方式
        self.sort_by = "accuracy"  # 默认按准确率排序
    
    def support(self, model_info: ModelInfoData) -> bool:
        # 这个可视化器需要至少包含准确率信息的模型
        return "final_test_acc" in model_info.metrics or "best_acc" in model_info.metrics
    
    def add_model_info(self, model_info: ModelInfoData):
        """添加要比较的模型信息"""
        self.model_infos.append(model_info)
    
    def set_sort_by(self, sort_by: str):
        """设置排序方式
        
        参数:
            sort_by: 排序依据，可选值: "accuracy" (按准确率) 或 "name" (按名称)
        """
        if sort_by in ["accuracy", "name"]:
            self.sort_by = sort_by
        else:
            logger.warning(f"无效的排序方式: {sort_by}，将使用默认值 'accuracy'")
    
    def visualize(self, model_info: ModelInfoData = None, show: bool = True, 
                  figsize: tuple = (15, 10), plot_type: str = "all") -> Optional[object]:
        """
        可视化模型准确率比较
        
        参数:
            model_info: 可选的模型信息对象
            show: 是否显示图表/表格
            figsize: 图表大小（表格模式下忽略）
            plot_type: 比较类型，可选值: "all" (全部), "bar" (柱状图), "ranking" (排名表格), "scatter" (散点图)
        
        返回:
            表格对象或None
        """
        try:
            # 如果传入了model_info，添加到比较列表
            if model_info:
                self.add_model_info(model_info)
            
            # 确保有模型可以比较
            if len(self.model_infos) < 2:
                logger.warning("模型比较需要至少2个模型信息")
                return None
            
            # 准备比较数据
            model_data = []
            
            for info in self.model_infos:
                # 获取模型名称或标识符，改进命名显示
                model_name = self._get_model_name(info)
                
                # 获取测试准确率，优先使用final_test_acc，其次是best_acc
                test_acc = info.metrics.get("final_test_acc", 0)
                if test_acc == 0:
                    test_acc = info.metrics.get("best_acc", 0)
                
                # 获取训练准确率
                train_acc = info.metrics.get("train_acc", 0)
                if train_acc == 0:
                    train_acc = info.metrics.get("final_train_acc", 0)
                
                # 获取验证准确率
                val_acc = info.metrics.get("val_acc", 0)
                if val_acc == 0:
                    val_acc = info.metrics.get("best_val_acc", 0)
                
                # 获取训练时间
                train_time = 0
                if "total_training_time" in info.metrics:
                    time_str = info.metrics["total_training_time"]
                    time_match = re.search(r'([\d.]+)', time_str)
                    if time_match:
                        train_time = float(time_match.group(1))
                
                # 获取参数量
                param_count = info.params.get("total_params", 0)
                
                model_data.append({
                    "name": model_name,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "train_time": train_time,
                    "params": param_count,
                    "path": info.path
                })
            
            # 根据排序方式排序
            if self.sort_by == "accuracy":
                model_data.sort(key=lambda x: x["test_acc"], reverse=True)
            else:
                model_data.sort(key=lambda x: x["name"])
            
            # 提取排序后的数据
            model_names = [d["name"] for d in model_data]
            test_accs = [d["test_acc"] for d in model_data]
            train_accs = [d["train_acc"] for d in model_data]
            val_accs = [d["val_acc"] for d in model_data]
            train_times = [d["train_time"] for d in model_data]
            params_counts = [max(1, d["params"]) for d in model_data]  # 确保至少为1，避免对数刻度问题
            
            # 计算准确率差异（泛化能力）
            acc_diffs = [test - train for test, train in zip(test_accs, train_accs)]
            
            # 仅实现ranking模式（表格输出），其他模式暂不实现
            if plot_type == "ranking":
                # 使用PrettyTable创建表格
                table = PrettyTable()
                
                # 设置表格标题
                print("\n" + "="*80)
                print("📊 模型准确率详细排名比较")
                print("="*80)
                
                # 设置表格字段
                table.field_names = ["排名", "模型名称", "测试准确率", "训练准确率", "验证准确率", "准确率差异", "参数量", "训练时间"]
                
                # 设置表格对齐方式
                table.align["模型名称"] = "l"  # 左对齐
                table.align["测试准确率"] = "r"
                table.align["训练准确率"] = "r"
                table.align["验证准确率"] = "r"
                table.align["准确率差异"] = "r"
                table.align["参数量"] = "r"
                table.align["训练时间"] = "r"
                
                # 添加数据行
                for i, data in enumerate(model_data, 1):
                    # 格式化参数量
                    params_formatted = f"{data['params']:,}" if data['params'] > 0 else "N/A"
                    
                    # 格式化训练时间
                    time_formatted = f"{data['train_time']:.1f}s" if data['train_time'] > 0 else "N/A"
                    
                    # 计算准确率差异
                    acc_diff = data['test_acc'] - data['train_acc']
                    acc_diff_formatted = f"{acc_diff:+.4f}"
                    
                    # 添加行数据
                    table.add_row([
                        i,
                        data['name'],
                        f"{data['test_acc']:.4f}",
                        f"{data['train_acc']:.4f}" if data['train_acc'] > 0 else "N/A",
                        f"{data['val_acc']:.4f}" if data['val_acc'] > 0 else "N/A",
                        acc_diff_formatted,
                        params_formatted,
                        time_formatted
                    ])
                
                # 设置表格样式
                table.border = True
                table.header = True
                table.padding_width = 1
                
                # 打印表格
                if show:
                    print(table)
                    
                    # 添加摘要统计信息
                    max_acc_idx = test_accs.index(max(test_accs))
                    min_acc_idx = test_accs.index(min(test_accs))
                    
                    print("\n" + "="*80)
                    print("📋 模型比较摘要")
                    print("="*80)
                    print(f"🏆 最佳模型: {model_data[max_acc_idx]['name']} (准确率: {test_accs[max_acc_idx]:.4f})")
                    print(f"📉 最差模型: {model_data[min_acc_idx]['name']} (准确率: {test_accs[min_acc_idx]:.4f})")
                    print(f"📊 平均准确率: {sum(test_accs) / len(test_accs):.4f}")
                    print(f"📏 准确率范围: {max(test_accs) - min(test_accs):.4f}")
                    print("="*80)
                
                return table
            else:
                # 其他模式的实现可以在后续添加
                logger.info(f"模式 '{plot_type}' 尚未实现表格输出")
                print(f"⚠️ 警告：'{plot_type}' 模式尚未实现表格输出，请使用 'ranking' 模式")
                return None
            
        except Exception as e:
            logger.error(f"绘制模型比较可视化失败: {str(e)}")
            print(f"❌ 模型比较过程出错: {str(e)}")
            return None
    
    def _get_model_name(self, model_info: ModelInfoData) -> str:
        """获取有意义的模型名称"""
        # 优先从metrics或params中获取模型名称
        if "model_name" in model_info.metrics:
            return model_info.metrics["model_name"]
        elif "model_type" in model_info.params:
            return model_info.params["model_type"]
        
        # 从路径中提取有意义的名称
        path_basename = os.path.basename(model_info.path)
        
        # 检查是否包含模型名称关键字
        model_types = ["LeNet", "AlexNet", "VGG", "ResNet", "GoogLeNet", "DenseNet", "MLP", "NIN"]
        for model_type in model_types:
            if model_type.lower() in path_basename.lower():
                # 尝试提取epoch或准确率信息
                epoch_match = re.search(r'epoch_(\d+)', path_basename, re.IGNORECASE)
                acc_match = re.search(r'acc_([\d.]+)', path_basename, re.IGNORECASE)
                
                name_parts = [model_type]
                if acc_match:
                    name_parts.append(f"{float(acc_match.group(1))*100:.1f}%")
                if epoch_match:
                    name_parts.append(f"E{epoch_match.group(1)}")
                
                return "-".join(name_parts)
        
        # 如果没有找到明显的模型类型，截取路径名
        if len(path_basename) > 15:
            return path_basename[:12] + '...'
        return path_basename

# 保持原有导入用于兼容其他代码
import matplotlib.pyplot as plt