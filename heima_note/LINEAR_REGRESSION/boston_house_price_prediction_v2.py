# 导入依赖包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data  # 特征数据
y = boston.target  # 目标变量（房价）

# 数据预处理
print("数据集形状:", X.shape)
print("特征名称:", X.columns.tolist())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征工程 - 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 模型预测
y_pred = model.predict(X_test_scaled)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估结果:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# 改进的可视化部分
plt.figure(figsize=(18, 12))

# 1. 散点图：预测值 vs 实际值（增强版）
plt.subplot(2, 2, 1)
# 使用更大的点和更高的透明度
scatter = plt.scatter(y_test, y_pred, c='blue', alpha=0.6, s=60, 
                     label=f'预测值 vs 实际值 (R²={r2:.4f})')
# 理想预测线
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='理想预测线')
plt.xlabel('实际房价 ($1000s)', fontsize=12)
plt.ylabel('预测房价 ($1000s)', fontsize=12)
plt.title('预测值与实际值对比', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 2. 误差分布图
plt.subplot(2, 2, 2)
errors = y_pred - y_test
sns.histplot(errors, kde=True, bins=20, color='green')
plt.axvline(x=0, color='red', linestyle='--', label='零误差')
plt.xlabel('预测误差 ($1000s)', fontsize=12)
plt.ylabel('频率', fontsize=12)
plt.title('预测误差分布', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 3. 实际值与预测值的折线对比（前50个样本）
plt.subplot(2, 2, 3)
sample_indices = np.arange(50)  # 取前50个样本
plt.plot(sample_indices, y_test[:50], 'bo-', label='实际值', alpha=0.7)
plt.plot(sample_indices, y_pred[:50], 'ro-', label='预测值', alpha=0.7)
plt.xlabel('样本索引', fontsize=12)
plt.ylabel('房价 ($1000s)', fontsize=12)
plt.title('前50个样本的实际值与预测值对比', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 4. 误差绝对值与实际值的关系
plt.subplot(2, 2, 4)
plt.scatter(y_test, np.abs(errors), c='purple', alpha=0.6, s=50)
plt.axhline(y=rmse, color='orange', linestyle='--', label=f'RMSE = {rmse:.2f}')
plt.xlabel('实际房价 ($1000s)', fontsize=12)
plt.ylabel('预测误差绝对值 ($1000s)', fontsize=12)
plt.title('误差绝对值与实际房价的关系', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# 特征重要性
coefficients = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_
})
coefficients = coefficients.sort_values(by='系数', ascending=False)
print("\n特征重要性（系数）:")
print(coefficients)

# 特征系数可视化
plt.figure(figsize=(12, 6))
sns.barplot(x='系数', y='特征', data=coefficients)
plt.title('各特征对房价的影响系数', fontsize=14)
plt.xlabel('系数值', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
