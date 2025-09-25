from typing import List
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import json
from typing import List, Optional, Dict, Any
from src.model_visualization.data_models import ModelInfoData, MetricData
from src.helper_utils.helper_tools_registry import ToolRegistry
from src.utils.log_utils import get_logger
from prettytable import PrettyTable

# 导入基类
from src.model_visualization.model_info_visualizers import BaseModelInfoVisualizer

# 初始化日志器
logger = get_logger(name=__name__, log_file="logs/model_info_visualizer.log", global_level="INFO")


class TimestrideModelComparisonVisualizer(BaseModelInfoVisualizer):
    """Timestride模型准确率比较可视化器：专注于比较timestride命名空间下多个模型的准确率指标"""
    
    def __init__(self):
        self.model_infos = []
        # 新增：设置排序方式
        self.sort_by = "accuracy"  # 默认按准确率排序
        self.namespace = "timestride"
    
    def support(self, model_info: ModelInfoData) -> bool:
        # 这个可视化器需要至少包含准确率信息的模型
        return model_info.namespace == self.namespace
    
    def add_model_info(self, model_info: ModelInfoData):
        """添加要比较的模型信息"""
        if self.support(model_info):
            self.model_infos.append(model_info)
        else:
            logger.warning(f"模型 {model_info.path} 不属于timestride命名空间，不添加到比较列表")
    
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
            plot_type: 比较类型，可选值: "all" (全部), "ranking" (排名表格)
        
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
                # 获取模型名称或标识符
                model_name = self._get_model_name(info)
                
                # 获取测试准确率
                test_acc = info.metrics.get("accuracy", 0)
                
                # 尝试从training_metrics.json获取更多指标
                train_acc = 0
                val_acc = 0
                train_loss = 0
                val_loss = 0
                epochs = 0
                
                # 从路径中解析一些参数信息
                model_path = info.path
                
                # 从配置参数中获取信息
                model_type = info.params.get("model", "Unknown")
                seq_len = info.params.get("seq_len", "Unknown")
                d_model = info.params.get("d_model", "Unknown")
                n_heads = info.params.get("n_heads", "Unknown")
                
                model_data.append({
                    "name": model_name,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "model_type": model_type,
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epochs": epochs,
                    "path": info.path
                })
            
            # 根据排序方式排序
            if self.sort_by == "accuracy":
                model_data.sort(key=lambda x: x["test_acc"], reverse=True)
            else:
                model_data.sort(key=lambda x: x["name"])
            
            # 实现ranking模式（表格输出）
            if plot_type == "ranking" or plot_type == "all":
                # 使用PrettyTable创建表格
                table = PrettyTable()
                
                # 设置表格标题
                print("\n" + "="*80)
                print("📊 Timestride模型准确率详细排名比较")
                print("="*80)
                
                # 设置表格字段
                table.field_names = ["排名", "模型名称", "模型类型", "测试准确率", "序列长度", "嵌入维度", "注意力头数", "来源路径"]
                
                # 设置表格对齐方式
                table.align["模型名称"] = "l"  # 左对齐
                table.align["模型类型"] = "l"
                table.align["测试准确率"] = "r"
                table.align["序列长度"] = "r"
                table.align["嵌入维度"] = "r"
                table.align["注意力头数"] = "r"
                
                # 添加数据行
                for i, data in enumerate(model_data, 1):
                    # 格式化路径（只显示最后两级目录）
                    path_parts = data['path'].split('/')[-3:]
                    short_path = '/'.join(path_parts)
                    
                    # 添加行数据
                    table.add_row([
                        i,
                        data['name'],
                        data['model_type'],
                        f"{data['test_acc']:.4f}",
                        data['seq_len'],
                        data['d_model'],
                        data['n_heads'],
                        short_path
                    ])
                
                # 设置表格样式
                table.border = True
                table.header = True
                table.padding_width = 1
                
                # 打印表格
                if show:
                    print(table)
                    
                    # 添加摘要统计信息
                    test_accs = [d["test_acc"] for d in model_data]
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
                logger.info(f"模式 '{plot_type}' 尚未实现表格输出")
                print(f"⚠️ 警告：'{plot_type}' 模式尚未实现表格输出，请使用 'ranking' 模式")
                return None
            
        except Exception as e:
            logger.error(f"绘制模型比较可视化失败: {str(e)}")
            print(f"❌ 模型比较过程出错: {str(e)}")
            return None
    
    def _get_model_name(self, model_info: ModelInfoData) -> str:
        """获取有意义的模型名称"""
        # 尝试从路径中提取模型名称
        path_basename = os.path.basename(model_info.path)
        
        # 检查是否包含模型名称关键字
        model_types = ["TimesNet", "Transformer", "LSTM", "GRU", "CNN"]
        for model_type in model_types:
            if model_type.lower() in path_basename.lower():
                # 尝试提取一些关键参数信息
                seq_len_match = re.search(r'sl(\d+)', path_basename)
                d_model_match = re.search(r'dm(\d+)', path_basename)
                
                name_parts = [model_type]
                if seq_len_match:
                    name_parts.append(f"sl{seq_len_match.group(1)}")
                if d_model_match:
                    name_parts.append(f"dm{d_model_match.group(1)}")
                
                return "-".join(name_parts)
        
        # 如果没有找到明显的模型类型，截取路径名
        if len(path_basename) > 20:
            return path_basename[:17] + '...'
        return path_basename
    
    def create_model_info_from_path(self, model_path: str) -> Optional[ModelInfoData]:
        """从模型路径创建ModelInfoData对象"""
        try:
            # 检查路径是否存在
            if not os.path.exists(model_path):
                logger.warning(f"模型路径不存在: {model_path}")
                return None
            
            # 初始化参数和指标
            params = {}
            metrics = {}
            
            # 尝试读取args.json
            args_path = os.path.join(model_path, "args.json")
            if os.path.exists(args_path):
                with open(args_path, 'r') as f:
                    params = json.load(f)
            
            # 尝试读取test_results/metrics.json
            test_metrics_path = os.path.join(model_path, "test_results", "metrics.json")
            if os.path.exists(test_metrics_path):
                with open(test_metrics_path, 'r') as f:
                    test_metrics = json.load(f)
                    metrics.update(test_metrics)
            
            # 尝试从training_metrics.json获取更多指标
            train_metrics_path = os.path.join(model_path, "training_metrics.json")
            if os.path.exists(train_metrics_path):
                with open(train_metrics_path, 'r') as f:
                    train_metrics = json.load(f)
                    # 获取最后一个epoch的指标
                    if train_metrics and isinstance(train_metrics, list) and len(train_metrics) > 0:
                        last_epoch = train_metrics[-1]
                        if "test_accuracy" in last_epoch:
                            metrics["accuracy"] = last_epoch["test_accuracy"]
            
            # 提取模型类型
            model_type = params.get("model", "Unknown")
            
            # 创建ModelInfoData对象
            model_info = ModelInfoData(
                type="model",
                path=model_path,
                model_type=model_type,
                params=params,
                metrics=metrics,
                timestamp=os.path.getmtime(model_path),
                namespace=self.namespace
            )
            
            return model_info
        except Exception as e:
            logger.error(f"从路径创建模型信息失败: {str(e)}")
            return None

# 注册可视化器到timestride命名空间
from src.model_visualization.model_info_visualizers import ModelInfoVisualizerRegistry
ModelInfoVisualizerRegistry.register(TimestrideModelComparisonVisualizer(), namespace="timestride")