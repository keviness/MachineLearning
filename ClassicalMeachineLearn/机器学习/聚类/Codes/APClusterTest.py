from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
import numpy as np

# 参数注释
# damping : 衰减系数，默认为5
# convergence_iter : 迭代次后聚类中心没有变化，算法结束，默认为
# max_iter : 最大迭代次数，默认
# copy : 是否在元数据上进行计算，默认True，在复制后的数据上进行计算。
# preference : S的对角线上的值
# affinity :S矩阵（相似度），默认为euclidean（欧氏距离）矩阵，即对传入的X计算距离矩阵，也可以设置为precomputed，那么X就作为相似度矩阵。

# 生成测试数据
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
print('X:\n',X)
print('labels_true:\n', labels_true)
# AP模型拟合
af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
print('cluster_centers_indices:\n', cluster_centers_indices)
labels = af.labels_
print('labels:\n', labels)

new_X = np.column_stack((X, labels))
n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
'''
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
print('Top 10 sapmles:', new_X[:10])
# 图形展示
import matplotlib.pyplot as plt
from itertools import cycle
plt.close('all')
plt.figure(1)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
'''