from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import torch.nn as nn
import torch.optim as optim

"""
核心是将基类从 “包含部分实现的工具类” 转变为 “纯接口 + 模板说明类”，只定义 “必须做什么” 和 “数据结构规范”，把 “具体怎么做” 完全交给子类，这样基类会更轻量化、更易扩展。

只保留抽象方法：明确子类必须实现的核心流程与组件，无任何具体代码实现（如文件写入、指标计算）。
用文档替代实现：通过类注释、方法注释、示例数据结构，告知子类需处理的文件（如metrics.json）和指标格式，而非用代码强制实现。
保留必要属性：仅定义核心状态属性（如模型、优化器、指标容器），不初始化具体值（由子类在__init__或_init_metrics中处理）
"""


class LightBaseTrainerTemplate(ABC):
    """纯模板化基类：仅定义接口规范与数据结构说明，无具体实现
    子类需严格遵循以下约定：
    1. 必须实现所有@abstractmethod装饰的方法
    2. 必须按规范处理4类文件：checkpoint.pth、config.json、metrics.json、epoch_metrics.json
       - config.json：单独保存训练配置信息
       - epoch_metrics.json：保存每个轮次的详细指标，为JSON数组格式
       - metrics.json：包含所有轮次指标列表及最终训练指标
    3. 指标数据结构需符合下方“指标规范说明”
    """

    def __init__(self, config: Dict[str, Any], save_dir: str):
        self.config = config  # 训练配置，最终需单独保存到config.json
        self.save_dir = save_dir  # 文件保存根目录（子类需自行创建目录）
        
        # 1. 核心组件（子类必须在_build_*方法中赋值）
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.loss_fn: Optional[nn.Module] = None
        self.train_loader = None
        self.test_loader = None

        # 2. 训练状态（子类需自行维护更新）
        self.epoch = 0  # 当前训练轮次
        self.best_metric = -float('inf')  # 最佳指标值（子类需定义判断逻辑）
        self.best_metric_name = ""  # 最佳指标名称（如"test_acc"，子类在_init_metrics中定义）
        self.best_epoch = 0  # 最佳指标对应的轮次

        # 3. 指标容器（子类需按“指标规范说明”填充）
        self.metrics: List[Dict[str, Any]] = []  # 所有轮次指标列表（最终存入metrics.json的epoch_metrics字段）
        self.current_epoch_metrics: Dict[str, Any] = {}  # 当前轮次指标（用于更新epoch_metrics.json）

        # 初始化指标规范（子类必须实现，明确指标名称与判断逻辑）
        self._init_metrics()

    # -------------------------- 一、核心组件构建接口（必须实现） --------------------------
    @abstractmethod
    def _build_model(self) -> None:
        """必须实现：构建模型并赋值给self.model
        示例：self.model = ResNet18(num_classes=config['data']['num_classes']).to(device)
        """
        pass

    @abstractmethod
    def _build_optimizer(self) -> None:
        """必须实现：构建优化器并赋值给self.optimizer
        示例：self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        """
        pass

    @abstractmethod
    def _build_scheduler(self) -> None:
        """必须实现：构建学习率调度器（可选赋值为None）
        示例：self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        """
        pass

    @abstractmethod
    def _build_loss_function(self) -> None:
        """必须实现：构建损失函数并赋值给self.loss_fn
        示例：self.loss_fn = nn.CrossEntropyLoss()
        """
        pass

    @abstractmethod
    def _build_data_loaders(self) -> None:
        """必须实现：构建数据加载器并赋值给self.train_loader/self.test_loader
        示例：self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        """
        pass

    @abstractmethod
    def _init_metrics(self) -> None:
        """必须实现：初始化指标规范
        要求：1. 赋值self.best_metric_name（如"test_acc"、"test_rmse"）
             2. 初始化self.best_metric（如损失类指标初始化为+inf，准确率初始化为- inf）
        示例：
            self.best_metric_name = "test_acc"
            self.best_metric = 0.0  # 准确率越大越好，初始值设为0
        """
        pass

    @abstractmethod
    def set_model_mode(self, mode: str) -> None:
        """设置模型模式（如"train"、"eval"）"""
        pass

    # -------------------------- 二、核心流程接口（必须实现） --------------------------
    @abstractmethod
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """必须实现：单轮训练逻辑
        返回值：当前轮次训练指标字典（key为指标名，如"loss"、"acc"）
        示例返回：{"loss": 0.23, "acc": 0.92}
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """必须实现：模型评估逻辑
        返回值：当前轮次评估指标字典（key为指标名，如"loss"、"acc"、"map"）
        示例返回：{"loss": 0.31, "acc": 0.89}
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """必须实现：完整训练流程
        要求：1. 循环调用train_one_epoch和evaluate
             2. 调用record_epoch_metrics记录指标
             3. 调用save_checkpoint保存模型
             4. 训练开始前保存config.json
             5. 每个epoch结束后更新epoch_metrics.json
             6. 训练结束后生成完整的metrics.json，包含：
                - 最终训练指标（如final_train_loss、final_train_acc、final_test_acc）
                - 最佳测试指标（best_test_acc）
                - 总训练时间信息（total_training_time、samples_per_second）
                - 所有轮次指标列表（epoch_metrics）
                - 训练开始和结束时间戳
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any, device: Optional[str] = None) -> Tuple[Any, ...]:
        """必须实现：模型推理逻辑
        返回值：推理结果（如分类任务返回(class_ids, scores)，回归任务返回(pred_values)）
        """
        pass

    # -------------------------- 三、文件与指标处理接口（必须实现） --------------------------
    @abstractmethod
    def record_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                           eval_metrics: Dict[str, float], epoch_time: float) -> bool:
        """必须实现：记录当前轮次指标（用于更新epoch_metrics.json和metrics容器）
        输入：train_metrics（train_one_epoch返回值）、eval_metrics（evaluate返回值）、epoch_time（本轮耗时，秒）
        返回值：是否为最佳模型（bool，用于决定是否保存best_model.pth）
        指标规范说明：
            1. current_epoch_metrics需包含的key：
               - "epoch"：轮次（int）
               - "train_loss"：训练损失（float）
               - "train_acc"：训练准确率（float）
               - "test_acc"：测试准确率（float）
               - "best_test_acc"：当前最佳测试准确率（float）
               - "epoch_time"：本轮耗时（float，秒）
            2. 需将current_epoch_metrics追加到metrics列表中（最终用于构建metrics.json的epoch_metrics字段）
            3. epoch_metrics.json需实时更新，保存所有已完成轮次的指标列表
        """
        pass

    @abstractmethod
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """必须实现：保存模型 checkpoint（pth文件）
        要求：1. 必须保存的key：
               - "epoch"：当前轮次
               - "model_state"：模型权重（self.model.state_dict()）
               - "best_metric"：当前最佳指标值
               - "best_metric_name"：最佳指标名称
               - "config"：训练配置（self.config）
             2. 可选保存的key：
               - "optimizer_state"：优化器状态（self.optimizer.state_dict()）
               - "scheduler_state"：调度器状态（self.scheduler.state_dict()）
             3. 若is_best=True，需额外保存为"best_model.pth"
             4. 同时需要确保：
               - 单独保存config.json文件
               - 实时更新epoch_metrics.json文件
               - 训练结束时生成完整的metrics.json文件
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """必须实现：加载模型 checkpoint（pth文件）
        要求：1. 加载权重到self.model、self.optimizer、self.scheduler（若存在）
             2. 恢复训练状态（self.epoch、self.best_metric、self.best_metric_name）
        返回值：加载的checkpoint字典
        """
        pass

    # -------------------------- 四、辅助流程（可选实现，建议保留） --------------------------
    def build_all_components(self) -> None:
        """统一组件构建入口（子类可直接调用，无需重写）
        作用：按固定顺序调用所有_build_*方法，确保流程一致
        """
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_loss_function()
        self._build_data_loaders()
        print(f"[{self.__class__.__name__}] 所有组件构建完成")