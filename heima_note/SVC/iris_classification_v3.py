import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 1. 生成非线性可分数据集（半月形数据）
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)  # 300个样本，添加少量噪声
print("数据集形状：特征", X.shape, "标签", y.shape)


# 2. 数据预处理
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 测试集占30%
)

# 标准化（SVM对特征尺度敏感，必须标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. 定义模型（高斯核SVM）
# 高斯核参数说明：
# - kernel='rbf'：指定使用高斯核（RBF核）
# - gamma：控制核函数的宽度（影响模型复杂度），gamma越大，核函数越窄，模型可能过拟合；gamma越小，核函数越宽，模型可能欠拟合
# - C：正则化参数，C越大，模型对训练集误差越敏感，可能过拟合；C越小，正则化越强，可能欠拟合
gamma_values = [0.1, 1, 10]  # 测试不同gamma值的效果
models = {f"gamma={g}": SVC(kernel='rbf', gamma=g, C=1.0, random_state=42) for g in gamma_values}


# 4. 训练模型并评估
for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} 模型评估：")
    print(f"准确率：{accuracy:.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred))


# 5. 可视化决策边界（对比不同gamma值的效果）
plt.figure(figsize=(15, 4))

# 生成网格点用于绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for i, (name, model) in enumerate(models.items()):
    plt.subplot(1, 3, i+1)
    
    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和背景
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # 绘制训练集和测试集样本
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='训练集')
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='white', s=100, label='测试集')
    
    plt.title(f"{name} 决策边界")
    plt.xlabel("特征1（标准化后）")
    plt.ylabel("特征2（标准化后）")
    plt.legend()

plt.tight_layout()
plt.show()