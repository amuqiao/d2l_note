import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载手写数字数据集
digits = load_digits()
X = digits.data  # 特征数据：64维向量
y = digits.target  # 标签：0-9

# 查看数据集信息
print(f"数据集特征形状: {X.shape}")
print(f"数字类别: {np.unique(y)}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割训练集和测试集时保留原始索引（关键修复）
indices = np.arange(len(X))  # 保存原始数据的索引
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, indices, test_size=0.3, random_state=42, stratify=y
)

# 初始化KNN分类器
knn = KNeighborsClassifier()

# 定义超参数搜索范围
param_grid = {
    'n_neighbors': np.arange(1, 16),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# 网格搜索
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

# 预测与评估
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 打印混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), 
            yticklabels=np.unique(y))
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('手写数字识别混淆矩阵')
plt.show()

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化部分测试样本（修复索引问题）
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # 使用原始索引获取图像（关键修复）
    original_idx = idx_test[i]  # 从保存的测试集索引中获取原始数据索引
    plt.imshow(digits.images[original_idx],  # 直接用原始索引访问图像
               cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f"预测: {y_pred[i]}\n真实: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
    