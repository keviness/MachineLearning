from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target  #iris数据集

#选择K个最好的特征，返回选择特征后的数据
skb = SelectKBest(chi2, k=3)
#print('features:\n', features)
# 调用fit方法
#skb=skb.fit(X,y)
X_new = skb.fit_transform(X, y)
print('X:\n', X)
print('X_new:\n', X_new)
#调用属性scores_，获得chi2返回的得分
print('scores_:\n', skb.scores_)
# 调用属性pvalues_ ，获得chi2返回的P值
print('pvalues_:\n', skb.pvalues_ )

#返回特征过滤后保留下的特征列索引
featuresIndex = skb.get_support(indices=True) 
print('featuresIndex:\n', featuresIndex)

'''
# 转换数据，得到特征过滤后保留下的特征数据集
x_new=skb.transform(X)
print('x_new:\n', x_new)
'''