# 葡萄酒数据集集成学习案例

## 1. 需求分析
# 本案例使用sklearn中的葡萄酒数据集，该数据集包含178个样本，
# 每个样本有13个特征（葡萄酒的化学成分），目标是将葡萄酒分为3个类别。
# 我们将分别使用单决策树和AdaBoost集成学习模型进行分类，
# 并比较两种模型的性能差异，展示集成学习的优势。

## 2. 导入依赖包
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def setup_matplotlib_font():
    """配置matplotlib中文显示
    
    该函数会根据不同操作系统设置合适的字体，确保中文能够正确显示。
    支持Windows、macOS和Linux系统。
    
    Example:
        >>> from src.helper_utils.font_utils import setup_matplotlib_font
        >>> setup_matplotlib_font()  # 调用后，matplotlib将正确显示中文
    """
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
    elif sys.platform.startswith("darwin"):
        plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC"]
    elif sys.platform.startswith("linux"):
        plt.rcParams["font.family"] = ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
    
setup_matplotlib_font()

## 3. 获取数据
# 加载葡萄酒数据集
wine = load_wine()
X = wine.data  # 特征数据
y = wine.target  # 目标标签

# 查看数据集信息
print(f"数据集特征形状: {X.shape}")
print(f"数据集标签形状: {y.shape}")
print(f"类别数量: {len(np.unique(y))}")
print(f"特征名称: {wine.feature_names}")

## 4. 数据预处理
# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 5. 实例化模型
# 单决策树模型
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)

# AdaBoost模型（以决策树为基分类器）
ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=50,  # 弱分类器数量
    learning_rate=1.0,  # 学习率
    random_state=42
)

## 6. 单决策树训练和评估
# 训练模型
dt_model.fit(X_train_scaled, y_train)

# 预测
y_pred_dt = dt_model.predict(X_test_scaled)

# 评估
print("\n===== 单决策树模型评估 =====")
print(f"准确率: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_dt))
print("\n分类报告:")
print(classification_report(y_test, y_pred_dt))

# 交叉验证
cv_scores_dt = cross_val_score(dt_model, X, y, cv=5)
print(f"\n交叉验证准确率: {cv_scores_dt.mean():.4f} ± {cv_scores_dt.std():.4f}")

## 7. AdaBoost训练和评估
# 训练模型
ada_model.fit(X_train_scaled, y_train)

# 预测
y_pred_ada = ada_model.predict(X_test_scaled)

# 评估
print("\n===== AdaBoost模型评估 =====")
print(f"准确率: {accuracy_score(y_test, y_pred_ada):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred_ada))
print("\n分类报告:")
print(classification_report(y_test, y_pred_ada))

# 交叉验证
cv_scores_ada = cross_val_score(ada_model, X, y, cv=5)
print(f"\n交叉验证准确率: {cv_scores_ada.mean():.4f} ± {cv_scores_ada.std():.4f}")

## 8. 模型比较可视化
plt.figure(figsize=(10, 6))
models = ['单决策树', 'AdaBoost']
cv_means = [cv_scores_dt.mean(), cv_scores_ada.mean()]
cv_stds = [cv_scores_dt.std(), cv_scores_ada.std()]

plt.bar(models, cv_means, yerr=cv_stds, capsize=10)
plt.ylim(0.8, 1.0)
plt.ylabel('交叉验证准确率')
plt.title('单决策树与AdaBoost模型性能比较')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 查看AdaBoost中每个特征的重要性
feature_importance = ada_model.feature_importances_
feature_names = wine.feature_names

# 排序并可视化特征重要性
indices = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), feature_importance[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title('AdaBoost模型特征重要性')
plt.tight_layout()
plt.show()
