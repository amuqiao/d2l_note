"""模型工具类"""
import torch
import os

class ModelUtils:
    """模型工具类：提供模型加载、保存等通用功能"""

    @staticmethod
    def load_model_weights(model, checkpoint_path, device=None):
        """加载模型权重

        参数:
            model: 模型实例
            checkpoint_path: 权重文件路径
            device: 设备（如'cuda'或'cpu'）

        返回:
            加载权重后的模型实例
        """
        # 确保路径规范化（处理相对路径和绝对路径）
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
        
        if not checkpoint_path.endswith('.pth'):
            raise ValueError(f"无效的模型文件格式: {checkpoint_path}，应为.pth文件")

        # 加载模型（添加版本兼容性处理）
        try:
            # 检查PyTorch版本是否支持weights_only参数
            try:
                # 尝试使用weights_only参数
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except TypeError:
                # 如果报错（不支持该参数），则不使用weights_only
                checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"加载模型文件失败: {str(e)}") from e

        # 验证checkpoint内容
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"模型文件格式错误，缺少'model_state_dict'键: {checkpoint_path}")

        # 加载权重
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @staticmethod
    def save_model(model, save_path, **kwargs):
        """保存模型权重和额外信息

        参数:
            model: 模型实例
            save_path: 保存路径
            **kwargs: 额外要保存的信息
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 构建checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            **kwargs
        }
        
        # 保存模型
        torch.save(checkpoint, save_path)
        return save_path