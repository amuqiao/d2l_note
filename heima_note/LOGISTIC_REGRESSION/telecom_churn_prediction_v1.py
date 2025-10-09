# 1. 导入依赖包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 2. 加载数据及数据预处理
# 使用IBM提供的电信客户流失数据集（多个镜像确保可靠性）
print("正在加载数据...")

# 定义缓存目录和文件路径
data_dir = "e:/github_project/d2l_note/data/telecom_churn"
cache_file = os.path.join(data_dir, "Telco-Customer-Churn.csv")

# 主数据源：IBM官方镜像（稳定可靠）
primary_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
# 备用数据源1
backup_url1 = "https://raw.githubusercontent.com/prakharrathi25/data-science-projects/master/Telco%20Customer%20Churn/Telco-Customer-Churn.csv"
# 备用数据源2
backup_url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00597/Telco-Customer-Churn.csv"

# 首先检查缓存文件是否存在
if os.path.exists(cache_file):
    print(f"从缓存文件加载数据: {cache_file}")
    df = pd.read_csv(cache_file)
else:
    # 尝试从远程下载数据，失败时自动切换到备用源
    try:
        print("正在从主数据源下载数据...")
        df = pd.read_csv(primary_url)
        print("使用主数据源加载成功")
    except:
        try:
            print("主数据源失败，正在从备用数据源1下载数据...")
            df = pd.read_csv(backup_url1)
            print("使用备用数据源1加载成功")
        except:
            print("主数据源和备用源1失败，正在从备用数据源2下载数据...")
            df = pd.read_csv(backup_url2)
            print("使用备用数据源2加载成功")
    
    # 创建缓存目录（如果不存在）
    os.makedirs(data_dir, exist_ok=True)
    # 保存数据到缓存文件
    df.to_csv(cache_file, index=False)
    print(f"数据已缓存到: {cache_file}")

# 移除不需要的ID列
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# 查看数据集基本信息
print("\n数据集形状:", df.shape)
print("数据集列名:", df.columns.tolist())
print("\n数据集前5行:")
print(df.head())

# 2.1 缺失值处理
print("\n缺失值情况：")
print(df.isnull().sum())

# 处理TotalCharges列可能存在的空值（该数据集已知问题）
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# 处理其他可能的缺失值
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2.2 确定特征值和目标值
# 明确指定目标列（IBM数据集标准列名为'Churn'）
target_col = 'Churn'
print(f"\n使用 '{target_col}' 作为目标列")

# 重命名目标列为统一的'churn'
df = df.rename(columns={target_col: 'churn'})

# 查看目标变量分布
print("\n目标变量分布:")
print(df['churn'].value_counts())

# 区分特征和目标变量
X = df.drop('churn', axis=1)
y = df['churn']

# 处理目标变量（将Yes/No转换为1/0）
y = y.map({'Yes': 1, 'No': 0})

# 处理分类特征
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 2.3 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 特征工程(标准化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 模型训练（逻辑回归）
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. 模型预测和评估
# 获取概率预测
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 使用自定义阈值0.6生成预测标签
y_pred_default = model.predict(X_test_scaled)  # 默认阈值0.5的预测结果
y_pred_custom = (y_prob >= 0.6).astype(int)  # 自定义阈值0.6的预测结果

# 计算并打印两种阈值下的准确率
accuracy_default = accuracy_score(y_test, y_pred_default)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print(f"\n默认阈值(0.5)准确率: {accuracy_default:.4f}")
print(f"自定义阈值(0.6)准确率: {accuracy_custom:.4f}")

# 使用自定义阈值的预测结果进行后续评估
y_pred = y_pred_custom

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(conf_matrix)

# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.show()

# 分类报告
class_report = classification_report(y_test, y_pred)
print("\n分类报告:")
print(class_report)

# ROC曲线和AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)')
plt.ylabel('真正例率 (TPR)')
plt.title('接收者操作特性曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='重要性', y='特征', data=feature_importance)
plt.title('特征重要性 (逻辑回归系数绝对值)')
plt.tight_layout()
plt.show()
                  