# 朴素贝叶斯（Naive Bayes）
## 一，朴素贝叶斯理论
### （一）贝叶斯理论核心思想
![NaiveBayes](./imgs/naiveBayes.png)
~~~
用p1(x,y) 表示数据点 (x,y) 属于类别1(图中红色圆点表示的类别)的概率，用 p2(x,y)表示数据点(x,y) 属于类别2(图中蓝色三角形表示的类别)的概率，那么对于一个新数据点(x,y)，可以用下面的规则来判断它的类别：
* 如果p1(x,y)>p2(x,y)，那么类别为1
* 如果p1(x,y)<p2(x,y)，那么类别为2
选择具有最高概率的决策，这就是贝叶斯决策理论的核心思想。
~~~

### （二）贝叶斯推断
![条件概率公式3](./imgs/条件概率公式3.png)
~~~
1，条件概率
* P(A)称为"先验概率"（Prior probability），即在B事件发生之前，我们对A事件概率的一个判断。
* P(A|B)称为"后验概率"（Posterior probability），即在B事件发生之后，我们对A事件概率的重新评估。
* P(B|A)/P(B)称为"可能性函数"（Likelyhood），这是一个调整因子，使得预估概率更接近真实概率。

所以，条件概率可以理解成下面的式子：
后验概率　＝　先验概率 ｘ 调整因子

2，贝叶斯推断的含义
我们先预估一个"先验概率"，然后加入实验结果，看这个实验到底是增强还是削弱了"先验概率"，由此得到更接近事实的"后验概率"。
* 如果"可能性函数"P(B|A)/P(B)>1，意味着"先验概率"被增强，事件A的发生的可能性变大。
* 如果"可能性函数"=1，意味着B事件无助于判断事件A的可能性；如果"可能性函数"<1，意味着"先验概率"被削弱，事件A的可能性变小。
~~~

### （三）朴素贝叶斯推断
朴素贝叶斯对条件个概率分布做了条件独立性的假设。 比如下面的公式，假设有n个特征：
![朴素贝叶斯推断公式1](./imgs/朴素贝叶斯推断公式1.png)
由于每个特征都是独立的，进一步拆分公式 ：
![朴素贝叶斯推断公式2](./imgs/朴素贝叶斯推断公式2.png)

## 二，朴素贝叶斯编程实现
~~~py
"""
函数说明:朴素贝叶斯分类器训练函数
Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非侮辱类的条件概率数组
	p1Vect - 侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)							#计算训练的文档数目
    numWords = len(trainMatrix[0])							#计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)		#文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)	#创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                        	#分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:							#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)							#取对数，防止下溢出          
    p0Vect = np.log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive							#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 非侮辱类的条件概率数组
	p1Vec -侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    	#对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
~~~

## 三，朴素贝叶斯sklearn实现
### （一）高斯朴素贝叶斯（Gaussian Naive Bayes）
#### 1，示例
~~~py
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(X, Y)
GaussianNB()
>>> print(clf.predict([[-0.8, -1]]))
[1]
>>> clf_pf = GaussianNB()
>>> clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB()
>>> print(clf_pf.predict([[-0.8, -1]]))
[1]
~~~

#### 2，常用方法
|Methods  | Descriptions|
|:-------:|:-----------:|
|fit(X, y[, sample_weight]) | Fit Gaussian Naive Bayes according to X, y|
|get_params([deep]) |Get parameters for this estimator.|
|partial_fit(X, y[, classes, sample_weight]) |Incremental fit on a batch of samples.|
|predict(X) | Perform classification on an array of test vectors X.|
|predict_log_proba(X) |Return log-probability estimates for the test vector X.|
|predict_proba(X) |Return probability estimates for the test vector X.|
|score(X, y[, sample_weight]) |Return the mean accuracy on the given test data and labels.|
|set_params(**params) |Set the parameters of this estimator.|

### （二）多项分布朴素贝叶斯（Multinomial Naive Bayes）
#### 1，示例
~~~py
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import MultinomialNB
>>> clf = MultinomialNB()
>>> clf.fit(X, y)
MultinomialNB()
>>> print(clf.predict(X[2:3]))
[3]
~~~

#### 2，方法
|Methods  | Descriptions|
|:-------:|:-----------:|
|fit(X, y[, sample_weight]) | Fit Naive Bayes classifier according to X, y|
|get_params([deep]) |Get parameters for this estimator.|
|partial_fit(X, y[, classes, sample_weight]) |Incremental fit on a batch of samples.|
|predict(X) | Perform classification on an array of test vectors X.|
|predict_log_proba(X) |Return log-probability estimates for the test vector X.|
|predict_proba(X) |Return probability estimates for the test vector X.|
|score(X, y[, sample_weight]) |Return the mean accuracy on the given test data and labels.|
|set_params(**params) |Set the parameters of this estimator.|

### （三）补充朴素贝叶斯（Complement Naive Bayes）
#### 1，示例
~~~py
import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import ComplementNB
>>> clf = ComplementNB()
>>> clf.fit(X, y)
ComplementNB()
>>> print(clf.predict(X[2:3])) #[3]
~~~
#### 2，方法
|Methods  | Descriptions|
|:-------:|:-----------:|
|fit(X, y[, sample_weight]) | Fit Naive Bayes classifier according to X, y|
|get_params([deep]) |Get parameters for this estimator.|
|partial_fit(X, y[, classes, sample_weight]) |Incremental fit on a batch of samples.|
|predict(X) | Perform classification on an array of test vectors X.|
|predict_log_proba(X) |Return log-probability estimates for the test vector X.|
|predict_proba(X) |Return probability estimates for the test vector X.|
|score(X, y[, sample_weight]) |Return the mean accuracy on the given test data and labels.|
|set_params(**params) |Set the parameters of this estimator.|

### （四）伯努利朴素贝叶斯（Bernoulli Naive Bayes）
#### 1，示例
~~~py
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> Y = np.array([1, 2, 3, 4, 4, 5])
>>> from sklearn.naive_bayes import BernoulliNB
>>> clf = BernoulliNB()
>>> clf.fit(X, Y)
BernoulliNB()
>>> print(clf.predict(X[2:3]))
[3]
~~~
#### 2，方法
|Methods  | Descriptions|
|:-------:|:-----------:|
|fit(X, y[, sample_weight]) | Fit Naive Bayes classifier according to X, y|
|get_params([deep]) |Get parameters for this estimator.|
|partial_fit(X, y[, classes, sample_weight]) |Incremental fit on a batch of samples.|
|predict(X) | Perform classification on an array of test vectors X.|
|predict_log_proba(X) |Return log-probability estimates for the test vector X.|
|predict_proba(X) |Return probability estimates for the test vector X.|
|score(X, y[, sample_weight]) |Return the mean accuracy on the given test data and labels.|
|set_params(**params) |Set the parameters of this estimator.|
