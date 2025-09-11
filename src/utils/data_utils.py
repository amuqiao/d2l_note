from d2l import torch as d2l

# ========================= 数据加载类 =========================
class DataLoader:
    """数据加载类（支持自定义批次大小和图像resize）"""

    @staticmethod
    def load_data(batch_size=256, resize=None):
        """
        加载Fashion-MNIST数据集
        Args:
            batch_size: 批次大小（默认256）
            resize: 图像resize尺寸（默认None，即28x28）
        Returns:
            train_iter: 训练数据迭代器
            test_iter: 测试数据迭代器
        """
        print(f"📥 加载Fashion-MNIST（batch_size={batch_size}, resize={resize}）")
        return d2l.load_data_fashion_mnist(batch_size=batch_size, resize=resize)