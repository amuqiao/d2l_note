import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable


# 设置matplotlib字体，确保中文能正确显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 1. 准备数据集（扩充后的数据集）
# 电影数据：[打斗次数, 接吻次数, 类型标签(0=动作片, 1=爱情片)]
# 扩充后的已知类型电影（增加了更多样本和一些边界案例）
movies = np.array([
    # 动作片 (标签0)
    [100, 10, 0], [95, 5, 0], [90, 20, 0], [85, 15, 0], [80, 30, 0],
    [75, 25, 0], [70, 40, 0], [65, 35, 0], [60, 20, 0], [55, 15, 0],
    [90, 10, 0], [85, 15, 0], [80, 10, 0], [75, 5, 0], [65, 10, 0],
    [50, 30, 0],  # 边界案例：打斗次数中等，接吻次数中等
    
    # 爱情片 (标签1)
    [60, 60, 1], [55, 65, 1], [50, 70, 1], [45, 75, 1], [40, 80, 1],
    [35, 85, 1], [30, 90, 1], [25, 95, 1], [20, 100, 1], [15, 90, 1],
    [30, 70, 1], [35, 65, 1], [40, 60, 1], [45, 55, 1], [50, 50, 1],
    [60, 40, 1]   # 边界案例：打斗次数中等，接吻次数中等
])

# 未知类型的测试电影
unknown_movies = np.array([
    [50, 50],  # 测试电影1：边界案例
    [75, 30],  # 测试电影2：偏动作片
    [35, 75],  # 测试电影3：偏爱情片
    [60, 40],  # 测试电影4：较难分类
    [45, 55],  # 测试电影5：较难分类
    [80, 20],  # 测试电影6：明显动作片
    [25, 80]   # 测试电影7：明显爱情片
])

print(f"扩充后的数据集包含 {len(movies)} 部电影，其中：")
print(f"动作片: {np.sum(movies[:, -1] == 0)} 部")
print(f"爱情片: {np.sum(movies[:, -1] == 1)} 部")

# 2. 划分特征和标签
X = movies[:, :-1]  # 特征：打斗次数和接吻次数
y = movies[:, -1]   # 标签：电影类型

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
unknown_movies_scaled = scaler.transform(unknown_movies)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 寻找最佳k值
print("\n寻找最佳k值...")
k_range = range(1, 21)  # 测试k=1到20（因数据集扩大，增加k的范围）
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 使用5折交叉验证计算平均准确率
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# 绘制k值与准确率的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o', color='b')
plt.xlabel('k值')
plt.ylabel('交叉验证平均准确率')
plt.title('k值与模型准确率的关系')
plt.grid(True)

# 找到最佳k值（准确率最高的k）
best_k = k_range[np.argmax(k_scores)]
print(f"最佳k值为: {best_k}，对应的准确率: {max(k_scores):.4f}")

# 保存k值选择图
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
plt.savefig(os.path.join(script_dir, 'k_value_selection.png'))

# 6. 使用最佳k值创建并训练KNN模型
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# 7. 模型评估
y_pred = knn.predict(X_test)
print("\n模型评估结果：")
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

# 8. 预测未知电影类型
predictions = knn.predict(unknown_movies_scaled)

# 9. 可视化结果
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
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
# 保存图片到脚本所在路径
plt.savefig(os.path.join(script_dir, 'movie_type_prediction.png'))

print("\n未知电影预测结果：")
for i, (fights, kisses) in enumerate(unknown_movies):
    print(f"电影{i+1}：打斗次数={fights}, 接吻次数={kisses}, 预测类型={movie_types[int(predictions[i])]}")
