import torch
from torch import nn

class LeNet(nn.Module):
    """LeNet卷积神经网络"""

    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # 卷积块2：Conv2d → Sigmoid → AvgPool2d
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),  # 10分类（Fashion-MNIST）
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = self.classifier(x)
        return x