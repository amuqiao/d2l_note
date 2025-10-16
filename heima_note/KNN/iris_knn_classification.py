# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

# 查看数据集信息
print(f"数据集特征: {iris.feature_names}")
print(f"数据集标签: {iris.target_names}")
print(f"数据集形状: {X.shape}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 初始化KNN分类器
knn = KNeighborsClassifier()

# 定义要搜索的参数网格
param_grid = {'n_neighbors': np.arange(1, 31)}

# 使用网格搜索结合交叉验证寻找最佳参数
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳交叉验证得分
print(f"最佳K值: {grid_search.best_params_['n_neighbors']}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 使用最佳参数的模型进行预测
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# 模型评估
print("\n测试集准确率:", accuracy_score(y_test, y_pred))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 交叉验证结果可视化
cv_results = grid_search.cv_results_
mean_test_scores = cv_results['mean_test_score']

plt.figure(figsize=(10, 6))
plt.plot(param_grid['n_neighbors'], mean_test_scores, marker='o')
plt.xlabel('K值 (n_neighbors)')
plt.ylabel('交叉验证准确率')
plt.title('不同K值对应的交叉验证准确率')
plt.xticks(param_grid['n_neighbors'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
