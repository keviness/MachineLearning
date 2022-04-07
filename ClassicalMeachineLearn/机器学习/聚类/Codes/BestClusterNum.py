def getBestClusterNumberKmeans(MACCSArray):
    '''
    K=range(1,6)
    sse_result=[]
    for k in K:
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(MACCSArray)
        sse_result.append(sum(np.min(cdist(MACCSArray,kmeans.cluster_centers_,'euclidean'),axis=1))/MACCSArray.shape[0])
    plt.plot(K,sse_result,'gx-')
    plt.xlabel('k')
    plt.ylabel(u'平均畸变程度')
    plt.title(u'肘部法则确定最佳的K值')
    plt.show()
'''
    K=range(2,7)
    score=[]
    for k in K:
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(MACCSArray)
        score.append(metrics.silhouette_score(MACCSArray,kmeans.labels_,metric='euclidean'))
    plt.plot(K,score,'r*-')
    plt.xlabel('k')
    plt.ylabel(u'轮廓系数')
    plt.title(u'轮廓系数确定最佳的K值')
    plt.show()

def getBestClusterNumber(MACCSArray):
    Scores = [0]  # 存放轮廓系数,根据轮廓系数的计算公式，只有一个类簇时，轮廓系数为0
    for k in range(2, 5):
        #estimator = AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit_predict(MACCSArray)
        estimator = KMeans(n_clusters=k).fit_predict(MACCSArray)
        Scores.append(metrics.silhouette_score(MACCSArray, estimator, metric='euclidean'))
    BestClusterNumber = Scores.index(max(Scores))+2
    print('BestClusterNumber:\n', BestClusterNumber)
    
    i = range(2, 6)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.plot(i,Scores,'g.-')
    # silhouette_score是绿色（数值越大越好） calinski_harabasz_score是蓝色（数值越大越好）
    plt.show()
    return BestClusterNumber