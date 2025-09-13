import os
import sys
import time
import json
import datetime
import argparse
import torch
import os
from typing import List, Dict, Any, Optional

# 导入日志工具
from src.utils.logger import info, warning, error, success, exception, init

init(logger_name=__name__)


# 解决OpenMP运行时库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有模型并注册
# 注意：这些导入会触发模型的自动注册
from src.models.lenet import LeNet, LeNetBatchNorm
from src.models.alexnet import AlexNet
from src.models.vgg import VGG
from src.models.nin import NIN
from src.models.googlenet import GoogLeNet
from src.models.resnet import ResNet
from src.models.dense_net import DenseNet
from src.models.mlp import MLP

from src.utils.model_registry import ModelRegistry
from src.trainer.trainer import Trainer
from src.utils.data_utils import DataLoader
from src.utils.visualization import VisualizationTool

class BatchTrainer:
    """批量训练工具类：用于批量训练所有已注册的模型"""
    
    def __init__(self, configs: Dict[str, Dict[str, Any]] = None, skip_models: List[str] = None):
        """
        初始化批量训练器
        
        Args:
            configs: 可选的模型配置字典，键为模型名称，值为配置参数
            skip_models: 可选的跳过的模型名称列表
        """
        self.configs = configs or {}
        self.skip_models = skip_models or []
        self.results = []
        
        # 设置中文字体
        VisualizationTool.setup_font()
    
    def get_all_registered_models(self) -> List[str]:
        """获取所有已注册且未被跳过的模型"""
        all_models = ModelRegistry.list_models()
        return [model for model in all_models if model not in self.skip_models]
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型的配置参数"""
        # 如果用户提供了配置，使用用户的配置
        if model_name in self.configs:
            return self.configs[model_name]
        
        # 否则使用模型注册中心的默认配置
        try:
            return ModelRegistry.get_config(model_name)
        except ValueError:
            # 如果没有默认配置，返回一个基础配置
            return {
                "num_epochs": 10,
                "lr": 0.01,
                "batch_size": 128,
                "resize": None,
                "input_size": (1, 1, 28, 28)  # 默认LeNet尺寸
            }
    
    def train_model(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        训练单个模型，直接使用优化后的训练器
        
        Args:
            model_name: 模型名称
            config: 训练配置参数
        
        Returns:
            训练结果字典（包含成功/失败状态和详细信息）
        """
        # 直接使用优化后的训练器执行训练
        enable_visualization = config.get("enable_visualization", False)
        save_every_epoch = config.get("save_every_epoch", False)
        
        # 创建训练器实例
        trainer = Trainer()
        
        # 使用训练器的run_training方法执行训练
        result = trainer.run_training(
            model_type=model_name,
            config=config,
            enable_visualization=enable_visualization,
            save_every_epoch=save_every_epoch
        )
        
        # 额外的保存目录信息（如果需要更详细的日志）
        if result["success"] and result.get("run_dir"):
            info(f"📁 训练结果保存目录: {result['run_dir']}")
        
        self.results.append(result)
        return result
    
    def train_all_models(self) -> List[Dict[str, Any]]:
        """训练所有已注册的模型，确保单个模型失败不影响整体流程"""
        models_to_train = self.get_all_registered_models()
        
        if not models_to_train:
            warning("⚠️ 没有找到可训练的模型！")
            return []
        
        info(f"📋 发现 {len(models_to_train)} 个可训练的模型：{', '.join(models_to_train)}")
        
        total_start_time = time.time()
        
        # 确保每个模型的训练过程都被独立保护，防止一个模型的异常影响整个循环
        for idx, model_name in enumerate(models_to_train, 1):
            try:
                info(f"\n📌 开始训练第 {idx}/{len(models_to_train)} 个模型: {model_name}")
                config = self.get_model_config(model_name)
                self.train_model(model_name, config)
                success(f"✅ 模型 {model_name} 训练处理完成")
            except Exception as e:
                # 这是额外的安全保障，防止train_model内部的异常处理失效
                error(f"❌ 模型 {model_name} 训练过程出现严重错误！")
                exception(f"💬 错误详情: {str(e)}")
                
                # 记录这个严重错误
                self.results.append({
                    "model_name": model_name,
                    "best_accuracy": 0.0,
                    "training_time": 0,
                    "error": f"严重异常: {str(e)}",
                    "config": config if 'config' in locals() else {},
                    "success": False
                })
                
                info(f"🔄 继续训练下一个模型...")
        
        total_time = time.time() - total_start_time
        info(f"\n{'='*60}")
        success(f"🏁 所有模型训练完成！总计耗时: {total_time:.2f}秒")
        info(f"{'='*60}")
        
        return self.results
    
    def save_results(self, output_path: str = None) -> str:
        """保存训练结果到指定路径的JSON文件
        
        Args:
            output_path: 结果保存路径，可以是文件名或目录名。如果是目录名，则会在该目录下创建带时间戳的文件名
                        默认保存在根目录下的log目录
                        
        Returns:
            保存结果的完整文件路径
        """
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 处理默认情况，设置为根目录下的log目录
        if not output_path:
            output_path = os.path.join(root_dir, "log")
            
        # 判断output_path是目录还是文件名
        if not os.path.splitext(output_path)[1]:
            # 没有文件扩展名，视为目录
            output_dir = output_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"batch_train_results_{timestamp}.json")
        else:
            # 有文件扩展名，视为完整文件路径
            output_file = output_path
            output_dir = os.path.dirname(os.path.abspath(output_file))
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存结果
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 批量训练结果已保存到: {output_file}")
        return output_file
    
    def print_summary(self):
        """打印训练结果摘要，增强失败模型的信息展示"""
        if not self.results:
            warning("⚠️ 没有训练结果可显示！")
            return
        
        info(f"\n{'='*120}")
        info("📊 批量训练结果摘要")
        info(f"{'='*120}")
        # 修改表格标题，添加开始和结束时间列
        info(f"{'模型名称':<15} {'最佳准确率':<12} {'训练时间(秒)':<12} {'开始时间':<20} {'结束时间':<20} {'状态'}")

        info(f"{'-'*120}")
        
        # 分开统计成功和失败的模型
        successful_models = []
        failed_models = []
        
        for result in self.results:
            status = "✅ 成功" if result["success"] else "❌ 失败"
            # 获取开始和结束时间，如果不存在则显示"-"
            start_time = result.get("training_start_time", "-")
            end_time = result.get("training_end_time", "-")
            # 打印包含时间信息的行，修正对齐格式
            info(f"{result['model_name']:<15} {result['best_accuracy']:<12.4f} {result['training_time']:<12.2f} {start_time:<20} {end_time:<20} {status}")
            info("")  # 在数据行之间增加一个空行
            
            if result["success"]:
                successful_models.append(result)
            else:
                failed_models.append(result)
        
        # 统计成功和失败的数量
        success_count = len(successful_models)
        fail_count = len(failed_models)
        
        info(f"{'-'*120}")
        info(f"总计: {success_count}个成功, {fail_count}个失败")
        if success_count > 0:
            avg_accuracy = sum(r["best_accuracy"] for r in successful_models) / success_count
            info(f"平均准确率: {avg_accuracy:.4f}")
        
        # 显示失败模型的详细错误信息
        if failed_models:
            info(f"\n{'='*60}")
            error("❌ 失败模型详情")
            info(f"{'='*60}")
            for failed_model in failed_models:
                error(f"\n模型名称: {failed_model['model_name']}")
                error(f"错误信息: {failed_model.get('error', '未知错误')}")
                info(f"{'='*60}")
        
        info(f"{'='*60}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量训练深度学习模型工具")
    # ["LeNet","AlexNet","VGG","NIN","GoogLeNet","ResNet","DenseNet"]
    parser.add_argument('--models', type=str, nargs='+', default=["LeNet"], help="指定要训练的模型名称列表，如 --models LeNet AlexNet")
    parser.add_argument('--skip', type=str, nargs='+', help="指定要跳过的模型名称列表，如 --skip VGG ResNet")
    parser.add_argument('--epochs', type=int, help="所有模型的训练轮数")
    parser.add_argument('--lr', type=float, help="所有模型的学习率")
    parser.add_argument('--batch_size', type=int, help="所有模型的批次大小")
    parser.add_argument('--output_dir', type=str, default='logs', help="训练结果输出目录路径")
    parser.add_argument('--enable_visualization', action='store_true', help="启用每个模型的训练可视化")
    
    return parser.parse_args()

def main():
    """主函数入口"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 构建配置字典
    configs = {}
    
    # 如果指定了特定的模型列表，创建这些模型的配置
    if args.models:
        for model_name in args.models:
            if ModelRegistry.is_registered(model_name):
                try:
                    config = ModelRegistry.get_config(model_name)
                    # 应用命令行参数的全局设置
                    if args.epochs is not None:
                        config["num_epochs"] = args.epochs
                    if args.lr is not None:
                        config["lr"] = args.lr
                    if args.batch_size is not None:
                        config["batch_size"] = args.batch_size
                    if args.enable_visualization:
                        config["enable_visualization"] = True
                    
                    configs[model_name] = config
                except ValueError:
                    warning(f"⚠️ 模型 '{model_name}' 没有默认配置，使用基础配置")
                    configs[model_name] = {
                        "num_epochs": args.epochs or 10,
                        "lr": args.lr or 0.01,
                        "batch_size": args.batch_size or 128,
                        "resize": None,
                        "input_size": (1, 1, 28, 28),  # 默认尺寸
                        "enable_visualization": args.enable_visualization
                    }
            else:
                error(f"❌ 模型 '{model_name}' 未注册，跳过")
    else:
        # 没有指定模型列表，使用命令行参数更新所有模型的默认配置
        if any([args.epochs is not None, args.lr is not None, args.batch_size is not None]):
            all_models = ModelRegistry.list_models()
            for model_name in all_models:
                if model_name not in (args.skip or []):
                    try:
                        config = ModelRegistry.get_config(model_name)
                        if args.epochs is not None:
                            config["num_epochs"] = args.epochs
                        if args.lr is not None:
                            config["lr"] = args.lr
                        if args.batch_size is not None:
                            config["batch_size"] = args.batch_size
                        if args.enable_visualization:
                            config["enable_visualization"] = True
                        
                        configs[model_name] = config
                    except ValueError:
                        pass  # 跳过没有默认配置的模型
    
    # 创建批量训练器并开始训练
    batch_trainer = BatchTrainer(configs=configs, skip_models=args.skip)
    
    # 如果指定了模型列表，只训练这些模型
    if args.models:
        valid_models = [m for m in args.models if ModelRegistry.is_registered(m)]
        if not valid_models:
            error(f"❌ 没有找到有效的模型！请检查 --models 参数")
        else:
            info(f"📋 开始训练指定的 {len(valid_models)} 个模型")
            # 确保单个模型失败不影响整个批量训练过程
            for idx, model_name in enumerate(valid_models, 1):
                try:
                    info(f"\n📌 开始训练第 {idx}/{len(valid_models)} 个模型: {model_name}")
                    config = batch_trainer.get_model_config(model_name)
                    batch_trainer.train_model(model_name, config)
                except Exception as e:
                    # 额外的安全保障层
                    error(f"❌ 模型 {model_name} 处理过程出现严重错误！")
                    exception(f"💬 错误详情: {str(e)}")
                    info(f"🔄 继续训练下一个模型...")
    else:
        # 训练所有模型
        batch_trainer.train_all_models()
    
    # 打印结果摘要
    batch_trainer.print_summary()
    
    # 保存结果到文件
    output_dir = args.output_dir
    batch_trainer.save_results(output_dir)

if __name__ == "__main__":
    main()