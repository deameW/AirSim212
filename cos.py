import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 读取CSV文件
data = pd.read_csv('generated_data.csv')

# 提取数据点的坐标
coordinates = data[['x', 'y', 'z']]

# 计算余弦相似度矩阵
cosine_sim = cosine_similarity(coordinates)

# 计算所有数据点之间的均值余弦相似度
mean_cosine_sim = np.mean(cosine_sim)

# 打印均值余弦相似度
print("均值余弦相似度:", mean_cosine_sim)
