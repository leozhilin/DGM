import pandas as pd
import numpy as np

file_path = 'GLM_Dataset.csv'
data = pd.read_csv(file_path)
# 确定前30%数据的数量
sample_size = int(0.3 * len(data))

# 从数据中选择前30%的样本
#sample_indices = np.arange(sample_size)
sample_indices = np.random.choice(data.index, size=sample_size, replace=False)

# 将这些样本的标签以0.5的概率选择为Negative或Positive
data.iloc[sample_indices, 1] = np.random.choice(['pos', 'neg'], size=sample_size, p=[0.5, 0.5])

# 保存修改后的数据到新的CSV文件
output_file_path = 'GLM_Change30_Dataset.csv'
data.to_csv(output_file_path, index=False, header=False)