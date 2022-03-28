'''
    凝聚层次算法：首先假定每个样本都是一个独立的聚类，如果统计出来的聚类数大于期望的聚类数，则从每个样本出发寻找离自己最近的另一个样本，
                与之聚集，形成更大的聚类，同时令总聚类数减少，不断重复以上过程，直到统计出来的聚类数达到期望值为止。

            凝聚层次算法的特点：
                1.聚类数k必须事先已知。借助某些评估指标，优选最好的聚类数。
                2.没有聚类中心的概念，因此只能在训练集中划分聚类，但不能对训练集以外的未知样本确定其聚类归属。不能预测。
                3.在确定被凝聚的样本时，除了以距离作为条件以外，还可以根据连续性来确定被聚集的样本。

            凝聚层次算法相关API：
                # 凝聚层次聚类器
                model = sc.AgglomerativeClustering(n_clusters=4)
                pred_y = model.fit_predict(x)   # 返回值为当前样本所属类别

    案例：重新加载multiple3.txt，使用凝聚层次算法进行聚类划分。

'''
import numpy as np
import matplotlib.pyplot as mp
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
# 读取数据，绘制图像
shopping_list = np.array([[1,0,1,1],
            [1,0,0,1],
            [1,0,1,1],
            [1,0,0,1],
            [1,0,1,0]])
print(shopping_list.shape)

# 基于Agglomerativeclustering完成聚类
model = AgglomerativeClustering(n_clusters=3)
pred_y = model.fit_predict(shopping_list)
print("pred_y",pred_y)
'''

Scores = [0]  # 存放轮廓系数,根据轮廓系数的计算公式，只有一个类簇时，轮廓系数为0
Scores1 = [0]   # 存放AH系数,根据轮廓系数的计算公式，只有一个类簇时，系数为0

A = np.array([
              [0, 1, 0, 1, 0, 1],
              [0, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1],
              [1, 0, 1, 0, 0, 1],
              [0, 0, 1, 0, 1, 0]
              ])
for k in range(2,6):
    #estimator = AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit_predict(A)
    estimator = KMeans(n_clusters=k).fit_predict(A)
    Scores.append(metrics.silhouette_score(A, estimator, metric='euclidean'))
    # 为Score列表添加质量度量值
    Scores1.append(metrics.calinski_harabasz_score(A, estimator))
print(Scores)
print(Scores1)
i = range(2, 7)
plt.xlabel('k')
plt.ylabel('value')
plt.plot(i,Scores,'g.-',i,Scores1,'b.-')
# silhouette_score是绿色（数值越大越好） calinski_harabasz_score是蓝色（数值越大越好）
plt.show()

'''
# 画图显示样本数据
mp.figure('Agglomerativeclustering', facecolor='lightgray')
mp.title('Agglomerativeclustering', fontsize=16)
mp.xlabel('X', fontsize=14)
mp.ylabel('Y', fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:, 0], x[:, 1], s=80, c=pred_y, cmap='brg', label='Samples')
mp.legend()
mp.show()
'''