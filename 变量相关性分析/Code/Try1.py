import seaborn as sns
data = sns.load_dataset('iris')
sampleDataFrame = data.iloc[:, :4] #取前四列数据
print('sampleData:\n', sampleDataFrame)

# Numpy
import numpy as np
X = sampleDataFrame['sepal_length']
Y = sampleDataFrame['petal_length']
result1 = np.corrcoef(X, Y)

# Scipy
import scipy.stats as ss
result3 = ss.pearsonr(X, Y)

# pandas
result4 = X.corr(Y)
result5 = sampleDataFrame.corr()

# 图可视化 sns
sns.pairplot(sampleDataFrame)
sns.pairplot(sampleDataFrame , hue ='sepal_width')

# pandas可视化
import pandas as pd
pd.plotting.scatter_matrix(sampleDataFrame, figsize=(12,12),range_padding=0.5)

# matplotlib
import matplotlib.pyplot as plt
figure, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(sampleDataFrame.corr(), square=True, annot=True, ax=ax)
