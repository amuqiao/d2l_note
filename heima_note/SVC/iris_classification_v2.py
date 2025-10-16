# 1. 导入依赖包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # 支持向量机分类器
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 2. 加载数据及数据预处理
# 2.1 加载数据
iris = load_iris()
X = iris.data  # 特征数据（4个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度）
y = iris.target  # 标签（0:山鸢尾, 1:变色鸢尾, 2:维吉尼亚鸢尾）
feature_names = iris.feature_names
target_names = iris.target_names

# 转换为DataFrame便于查看
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]
print("数据集前5行：")
print(df.head())


# 2.2 拆分训练集和测试集（避免数据泄露）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 测试集占20%，固定随机种子保证结果可复现
)


# 2.3 数据标准化（PCA对特征尺度敏感，需先标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 拟合训练集并标准化
X_test_scaled = scaler.transform(X_test)  # 用训练集的参数标准化测试集


# 2.4 数据降维（PCA降维到2维，便于可视化）
pca = PCA(n_components=2)  # 保留2个主成分
X_train_pca = pca.fit_transform(X_train_scaled)  # 拟合训练集并降维
X_test_pca = pca.transform(X_test_scaled)  # 用训练集的PCA参数降维测试集

# 打印PCA解释方差比例（评估降维效果）
print("\nPCA主成分解释方差比例：", pca.explained_variance_ratio_)
print("累计解释方差比例：", sum(pca.explained_variance_ratio_))


# 3. 模型训练（使用降维后的特征训练SVM模型）
model = SVC(kernel='rbf', random_state=42)  # 径向基核函数SVM，适合非线性分类
model.fit(X_train_pca, y_train)


# 4. 模型预测
y_pred = model.predict(X_test_pca)  # 对测试集进行预测


# 5. 模型评估
# 5.1 准确率及详细评估指标
print("\n模型评估结果：")
print(f"准确率：{accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=target_names))


# 5.2 可视化
plt.figure(figsize=(15, 6))

# 子图1：混淆矩阵
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, 
    annot=True,  # 显示数值
    fmt='d',  # 整数格式
    cmap='Blues', 
    xticklabels=target_names, 
    yticklabels=target_names
)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('混淆矩阵')

# 子图2：决策边界及样本分布
plt.subplot(1, 2, 2)
# 生成网格点用于绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# 绘制训练集和测试集样本
plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1], 
    c=y_train, cmap=plt.cm.coolwarm, 
    edgecolors='k', label='训练集'
)
plt.scatter(
    X_test_pca[:, 0], X_test_pca[:, 1], 
    c=y_test, cmap=plt.cm.coolwarm, 
    edgecolors='white', s=100, label='测试集'
)

plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维后的决策边界与样本分布')
plt.legend()

plt.tight_layout()
plt.show()