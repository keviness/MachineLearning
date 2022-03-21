# -*- coding:utf-8 -*-
  
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import euclidean_distances
  
#聚类算法之AP算法:
#1--与其他聚类算法不同,AP聚类不需要指定K(金典的K-Means)或者是其他描述聚类个数的参数
#2--一个聚类中最具代表性的点在AP算法中叫做Examplar,与其他算法中的聚类中心不同,examplar
#是原始数据中确切存在的一个数据点,而不是由多个数据点求平均而得到的聚类中心
#3--多次执行AP聚类算法,得到的结果完全一样的，即不需要进行随机选取初值步骤.
#算法复杂度较高,为O(N*N*logN),而K-Means只是O(N*K)的复杂度，当N》3000时,需要算很久
#AP算法相对于Kmeans优势是不需要指定聚类数量,对初始值不敏感
#AP算法应用场景：图像、文本、生物信息学、人脸识别、基因发现、搜索最优航线、 码书设计以及实物图像识别等领域
  
#算法详解: http://blog.csdn.net/helloeveryon/article/details/51259459
  
if __name__=='__main__':
#scikit中的make_blobs方法常被用来生成聚类算法的测试数据，直观地说，make_blobs会根据用户指定的特征数量、
# 中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
#函数原型：
# sklearn.datasets.make_blobs(n_samples=100, n_features=2,
     # centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)[source]
     #参数解析：
     # n_samples是待生成的样本的总数。
     # n_features是每个样本的特征数。
     #
     # centers表示类别数。
     #
     # cluster_std表示每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0, 3.0]。  
     N=400
     centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
     #生成聚类算法的测试数据
     data,y=ds.make_blobs(N,n_features=2,centers=centers,cluster_std=[0.5, 0.25, 0.7, 0.5],random_state=0)
     #计算向量之间的距离
     m=euclidean_distances(data,squared=True)
     #求中位数
     preference=-np.median(m)
     print('Preference:',preference)
  
     matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
     matplotlib.rcParams['axes.unicode_minus'] = False
     plt.figure(figsize=(12,9),facecolor='w')
     for i,mul in enumerate(np.linspace(1,4,9)):#遍历等差数列
         print('mul=',mul)
         p=mul*preference
         model=AffinityPropagation(affinity='euclidean',preference=p)
         af=model.fit(data)
         center_indices=af.cluster_centers_indices_
         n_clusters=len(center_indices)
         print ('p=%.1f'%mul,p,'聚类簇的个数为:',n_clusters)
         y_hat=af.labels_
  
         plt.subplot(3,3,i+1)
         plt.title(u'Preference：%.2f，簇个数：%d' % (p, n_clusters))
         clrs=[]
         for c in np.linspace(16711680, 255, n_clusters):
             clrs.append('#%06x' % c)
             for k, clr in enumerate(clrs):
                 cur = (y_hat == k)
                 plt.scatter(data[cur, 0], data[cur, 1], c=clr, edgecolors='none')
                 center = data[center_indices[k]]
                 for x in data[cur]:
                     plt.plot([x[0], center[0]], [x[1], center[1]], color=clr, zorder=1)
             plt.scatter(data[center_indices, 0], data[center_indices, 1], s=100, c=clrs, marker='*', edgecolors='k',
                         zorder=2)
             plt.grid(True)
         plt.tight_layout()
         plt.suptitle(u'AP聚类', fontsize=20)
         plt.subplots_adjust(top=0.92)
         plt.show()