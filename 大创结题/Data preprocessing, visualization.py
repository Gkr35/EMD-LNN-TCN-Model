import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
file_path = r'D:\桌面\大创论文\data.csv'
data = pd.read_csv(file_path)

# 确保 'DATATIME' 列为日期格式
data['DATATIME'] = pd.to_datetime(data['DATATIME'])

# 将 'DATATIME' 列设置为索引
data.set_index('DATATIME', inplace=True)

# 查看数据的前几行，确保DATATIME被正确设置为索引
print(data.head())

# 绘制所有特征的时间序列图
plt.figure(figsize=(16, 20))

# 特征名称列表
features = [
    'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 
    'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15'
]

for i, feature in enumerate(features):
    plt.subplot(5, 2, i + 1)  # 创建子图，最多10个
    plt.plot(data.index, data[feature], label=feature, color='blue')
    plt.title(f'{feature} Over Time')
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.legend()

plt.tight_layout()
plt.show()

# 计算相关性矩阵
correlation_matrix = data.corr()

# 绘制热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# 绘制所有特征分布图
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[feature], label=feature, color='purple')
    plt.title(f'{feature} Distribution Over Time')
    plt.xlabel('Time')
    plt.ylabel(feature)
    plt.legend()
    plt.show()
