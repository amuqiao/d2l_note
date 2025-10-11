# 1. 需求分析
"""
鸢尾花分类是机器学习中的经典多类别分类问题。
任务：根据鸢尾花的4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），
将其分为3个不同的品种：山鸢尾(setosa)、变色鸢尾(versicolor)和维吉尼亚鸢尾(virginica)。
本案例将使用LinearSVC（线性支持向量分类器）来构建分类模型，并评估其性能。
"""

# 2. 导入依赖包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 3. 获取数据
# 加载sklearn内置的鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据，包含4个特征
y = iris.target  # 标签数据，0、1、2分别代表三个品种
feature_names = iris.feature_names  # 特征名称
target_names = iris.target_names  # 类别名称

# 打印数据集基本信息
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"数据集规模: 样本数={X.shape[0]}, 特征数={X.shape[1]}")

# 4. 数据预处理
# 分割训练集和测试集，测试集占30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify确保分层抽样，保持类别比例
)

# 特征标准化（SVM对特征尺度敏感，标准化很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 用训练集拟合并转换
X_test_scaled = scaler.transform(X_test)  # 用训练集的参数转换测试集

# 5. 模型训练
# 创建LinearSVC模型
svc = LinearSVC(random_state=42, max_iter=10000)  # 增加max_iter确保收敛
# 训练模型
svc.fit(X_train_scaled, y_train)

# 6. 模型预测
# 在测试集上进行预测
y_pred = svc.predict(X_test_scaled)

# 7. 模型评估
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f}")

# 打印详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 打印混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("混淆矩阵解释: 行表示真实类别，列表示预测类别")

# 8. 展示分类效果
# 使用PCA将4维特征降维到2维以便可视化
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 创建网格以绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 对网格点进行预测
Z = svc.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# 绘制决策边界和数据点
plt.figure(figsize=(10, 6))
# 绘制决策区域
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# 绘制训练集数据点
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', 
           cmap=plt.cm.Paired, label='训练集')
# 绘制测试集预测结果
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, marker='x', s=100,
           cmap=plt.cm.Paired, label='测试集预测')
plt.xlabel('PCA特征1')
plt.ylabel('PCA特征2')
plt.title('LinearSVC鸢尾花分类结果 (PCA降维可视化)')
plt.legend()
plt.show()
