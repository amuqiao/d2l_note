# 导入依赖包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
# 注意：波士顿房价数据集在sklearn新版本中已移除，使用openml获取
boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data  # 特征数据
y = boston.target  # 目标变量（房价）

# 数据预处理 - 查看数据基本信息
print("数据集形状:", X.shape)
print("特征名称:", X.columns.tolist())
print("前5行数据:\n", X.head())

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 特征工程 - 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 对训练集拟合并标准化
X_test_scaled = scaler.transform(X_test)        # 使用训练集的标准化参数对测试集处理

# 模型训练 - 线性回归
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

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='预测值 vs 实际值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='理想预测线')
plt.xlabel('实际房价 ($1000s)')
plt.ylabel('预测房价 ($1000s)')
plt.title('波士顿房价预测：实际值 vs 预测值')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 输出特征系数（展示各特征对房价的影响）
coefficients = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_
})
coefficients = coefficients.sort_values(by='系数', ascending=False)
print("\n特征重要性（系数）:")
print(coefficients)
