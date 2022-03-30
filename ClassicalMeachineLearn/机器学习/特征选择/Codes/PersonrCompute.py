import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_regression
from scipy.stats import pearsonr
# -----------------------------
# |r|<0.3 不存在线性关系
# 0.3<|r|<0.5  低度线性关系
# 0.5<|r|<0.8  显著线性关系
# |r|>0.8  高度线性关系
# ------------------------------


data1 = pd.DataFrame(np.random.randn(200,4)*100, columns = ['A','B','C','D'])
# 相关性计算
print('data1:\n',data1.corr())

'''
data2 = pd.DataFrame(np.random.randn(200,2)*100, columns=['X','Y'])
X = data2.X
y = data2.Y
print('X:\n', X)
print('y:\n', y)
r,p = stats.pearsonr(X,y)  # 相关系数和P值
print('相关系数r为 = %6.3f，p值为 = %6.3f'%(r,p))
print('data2', data2.corr())
'''

X,y = make_regression(n_samples=1000, n_features=3, n_informative=1, noise=100, random_state=9527)
print('X:\n', X)
print('y:\n', y)
p1 = pearsonr(X[:,0],y)
p2 = pearsonr(X[:,1],y)
p3 = pearsonr(X[:,2],y)
print('p1:\n', p1)

print('p2:\n', p2)

print('p3:\n', p3)

