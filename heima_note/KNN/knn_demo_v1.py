import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 1. 准备数据集
# 电影数据：[打斗次数, 接吻次数, 类型标签(0=动作片, 1=爱情片)]
# 已知类型的电影
movies = np.array([
    [100, 10, 0],  # 动作片1
    [90, 20, 0],   # 动作片2
    [80, 30, 0],   # 动作片3
    [70, 40, 0],   # 动作片4
    [60, 60, 1],   # 爱情片1
    [50, 70, 1],   # 爱情片2
    [40, 80, 1],   # 爱情片3
    [30, 90, 1],   # 爱情片4
    [20, 100, 1],  # 爱情片5
    [30, 70, 1],   # 爱情片6
    [90, 10, 0],   # 动作片5
    [85, 15, 0]    # 动作片6
])

# 未知类型的测试电影
unknown_movies = np.array([
    [50, 50],  # 测试电影1
    [75, 30],  # 测试电影2
    [35, 75]   # 测试电影3
])

# 2. 划分特征和标签
X = movies[:, :-1]  # 特征：打斗次数和接吻次数
y = movies[:, -1]   # 标签：电影类型

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
unknown_movies_scaled = scaler.transform(unknown_movies)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 创建并训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)  # 选择k=3
knn.fit(X_train, y_train)

# 6. 模型评估
y_pred = knn.predict(X_test)
print("模型评估结果：")
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))
print("分类报告：")
print(classification_report(y_test, y_pred))

# 使用PrettyTable美化分类报告输出
print("\n分类报告（美化版）：")
# 获取分类报告数据（以字典形式）
report_dict = classification_report(y_test, y_pred, output_dict=True)

# 创建表格
report_table = PrettyTable()
report_table.field_names = ["类别", "精确率", "召回率", "F1分数", "支持样本数"]

# 添加每个类别的数据
for class_name, metrics in report_dict.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        class_label = '动作片' if class_name == '0' else '爱情片'
        report_table.add_row([
            class_label,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            metrics['support']
        ])

# 添加平均指标
report_table.add_row(["宏平均", 
                      f"{report_dict['macro avg']['precision']:.2f}",
                      f"{report_dict['macro avg']['recall']:.2f}",
                      f"{report_dict['macro avg']['f1-score']:.2f}",
                      report_dict['macro avg']['support']])
                        
report_table.add_row(["加权平均", 
                      f"{report_dict['weighted avg']['precision']:.2f}",
                      f"{report_dict['weighted avg']['recall']:.2f}",
                      f"{report_dict['weighted avg']['f1-score']:.2f}",
                      report_dict['weighted avg']['support']])

# 设置表格样式
report_table.align["类别"] = "l"
report_table.align["精确率"] = "r"
report_table.align["召回率"] = "r"
report_table.align["F1分数"] = "r"
report_table.align["支持样本数"] = "r"

# 打印表格
print(report_table)
print(f"\n准确率: {report_dict['accuracy']:.4f}")

# 7. 预测未知电影类型
predictions = knn.predict(unknown_movies_scaled)

# 8. 可视化结果
# 绘制已知电影散点图
plt.figure(figsize=(10, 6))

# 绘制训练数据点
action_movies = movies[y == 0]
romance_movies = movies[y == 1]

plt.scatter(action_movies[:, 0], action_movies[:, 1], color='red', marker='o', label='动作片')
plt.scatter(romance_movies[:, 0], romance_movies[:, 1], color='blue', marker='s', label='爱情片')

# 绘制未知电影点
plt.scatter(unknown_movies[:, 0], unknown_movies[:, 1], color='green', marker='*', s=200, label='未知电影')

# 添加标签和标题
plt.xlabel('打斗次数')
plt.ylabel('接吻次数')
plt.title('KNN算法预测电影类型')
plt.legend()

# 为未知电影添加预测结果文本
movie_types = ['动作片', '爱情片']
for i, (x, y) in enumerate(unknown_movies):
    plt.annotate(f'预测: {movie_types[int(predictions[i])]}', 
                 (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)

plt.grid(True)
# 获取脚本所在路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 保存图片到脚本所在路径
plt.savefig(os.path.join(script_dir, 'movie_type_prediction.png'))
print("\n未知电影预测结果：")
for i, (fights, kisses) in enumerate(unknown_movies):
    print(f"电影{i+1}：打斗次数={fights}, 接吻次数={kisses}, 预测类型={movie_types[int(predictions[i])]}")

# 9. 简单的KNN算法实现（为了理解原理）
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            # 计算与所有训练样本的距离
            distances = np.sqrt(np.sum((self.X_train - x) **2, axis=1))
            # 获取k个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            # 获取k个最近邻的标签
            k_nearest_labels = self.y_train[k_indices]
            # 投票决定最终类别
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)

# 使用自定义KNN实现
print("\n使用自定义KNN实现：")
simple_knn = SimpleKNN(k=3)
simple_knn.fit(X_train, y_train)
simple_predictions = simple_knn.predict(unknown_movies_scaled)

for i, (fights, kisses) in enumerate(unknown_movies):
    print(f"电影{i+1}：打斗次数={fights}, 接吻次数={kisses}, 预测类型={movie_types[int(simple_predictions[i])]}")