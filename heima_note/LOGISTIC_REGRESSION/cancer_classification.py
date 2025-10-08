# 导入依赖包
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 加载数据
cancer = load_breast_cancer()

# 转换为DataFrame以便查看
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# 2.1 缺失值处理
# 检查是否有缺失值
print("缺失值情况：")
print(df.isnull().sum())

# 乳腺癌数据集通常没有缺失值，如果有缺失值可以使用以下方法处理
# df = df.fillna(df.mean())  # 均值填充
# 或使用df.dropna(inplace=True) 删除缺失值

# 2.2 确定特征值和目标值
X = cancer.data  # 特征值
y = cancer.target  # 目标值，0表示恶性，1表示良性

# 查看数据集信息
print("\n数据集形状：", X.shape)
print("类别分布：", np.bincount(y))
print("类别含义：", cancer.target_names)

# 2.3 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 特征工程：标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练：逻辑回归
log_reg = LogisticRegression(max_iter=10000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 模型预测
y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)  # 预测概率

# 模型评估
print("\n===== 模型评估结果 =====")
# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 精确率
precision = precision_score(y_test, y_pred)
print(f"精确率: {precision:.4f}")

# 召回率
recall = recall_score(y_test, y_pred)
print(f"召回率: {recall:.4f}")

# F1分数
f1 = f1_score(y_test, y_pred)
print(f"F1分数: {f1:.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n混淆矩阵解释:")
print(f"真阴性: {cm[0,0]} (实际恶性，预测恶性)")
print(f"假阳性: {cm[0,1]} (实际恶性，预测良性)")
print(f"假阴性: {cm[1,0]} (实际良性，预测恶性)")
print(f"真阳性: {cm[1,1]} (实际良性，预测良性)")

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 查看特征重要性
feature_importance = pd.DataFrame({
    '特征': cancer.feature_names,
    '系数': log_reg.coef_[0]
})
feature_importance['绝对值'] = np.abs(feature_importance['系数'])
feature_importance = feature_importance.sort_values('绝对值', ascending=False)

print("\n最重要的5个特征:")
print(feature_importance.head(5))
