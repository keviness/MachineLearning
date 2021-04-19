## 逻辑回归（Logistic Regression）
## 一，Logistic回归算法思想
* Logistic回归是一种二分类算法那，它利用的是Sigmoid函数阈值在[0,1]这个特性。
* Logistic回归进行分类的主要思想是：根据现有数据对分类边界线建立回归公式，以此进行分类。
* Logistic本质上是一个基于条件概率的判别模型(Discriminative Model)。

### （一）Sigmoid函数
![Logistic Regression 2](./imgs/LogisticRegression2.png)
![Logistic Regression 1](./imgs/LogisticRegression1.jpeg)
* python实现Sigmoid函数
~~~py
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
~~~

### （二）代价函数
* 代价函数，是对于一个样本而言的。给定一个样本，我们就可以通过这个代价函数求出，样本所属类别的概率，而这个概率越大越好，所以也就是求解这个代价函数的最大值。
![Cost Function1](./imgs/CostFunction1.png)

* 满足J(θ)的最大的θ值即是我们需要求解的模型。
![Cost Function2](./imgs/CostFunction2.png)

### （三）梯度上升算法
![梯度上升](./imgs/梯度上升.png)
* python实现梯度上升算法
~~~py
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)         #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()   #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)        #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                  #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                        #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)         #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                 #将矩阵转换为数组，返回权重数组
~~~
* 2，python实现随机梯度算法
~~~py
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化										#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 															#返回
~~~

## 二，Logistic回归的一般过程：
* 收集数据：采用任意方法收集数据。
* 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
* 分析数据：采用任意方法对数据进行分析。
* 训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
* 测试算法：一旦训练步骤完成，分类将会很快。
* 使用算法：首先，我们需要输入一些数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数，就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；在这之后，就可以在输出的类别上做一些其他分析工作。
* 其他：Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法完成。

## 三，python实现Logistic回归
~~~py
# Sigmoid函数
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

# 分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 随机梯度算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   								#参数初始化										#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 	#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))	#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))	#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 	

def colicTest():
	frTrain = open('horseColicTraining.txt')	 #打开训练集
	frTest = open('horseColicTest.txt')			 #打开测试集
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)	#使用改进的随即上升梯度训练
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec) * 100 		#错误率计算
	print("测试集错误率为: %.2f%%" % errorRate)
~~~

## 四，sklearn module实现Logistic Regression
### （一）定义Logistic Regression
~~~py
sklearn.linear_model.LogisticRegression(
    penalty='l2', *,  # 用于指定惩罚中使用的规范。'newton-cg'，'sag'和'lbfgs'求解器仅支持l2罚分。仅“ saga”求解器支持“ elasticnet”。如果为“ none”（liblinear求解器不支持），则不应用任何正则化。
    dual=False, 
    tol=0.0001, 
    C=1.0, 
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight=None, # 与形式的类有关的权重。如果未给出，则所有类均应具有权重一。
    random_state=None,  # 在solver=='sag'，'saga'或'liblinear'时，用于随机整理数据
    solver='lbfgs', 
    max_iter=100, 
    multi_class='auto', 
    verbose=0, 
    warm_start=False, 
    n_jobs=None, 
    l1_ratio=None)
~~~
### （三）示例
~~~py
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegression(random_state=0).fit(X, y)
>>> clf.predict(X[:2, :])
array([0, 0])
>>> clf.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
>>> clf.score(X, y)
0.97...
~~~
### （四）方法
| Method |  Description  |
|:------:|:-------------:|
|decision_function（X） ｜ 预测样本的置信度得分|
|densify（）|将系数矩阵转换为密集数组格式|
|fit（X，y [，sample_weight]）|根据给定的训练数据拟合模型|
|get_params（[深的]）|获取此估计量的参数|
|predict（X）| 预测X中样本的类别标签|
|predict_log_proba（X）|预测概率估计的对数|
|predict_proba（X）| 概率估计|
|score（X，y [，sample_weight]）| 返回给定测试数据和标签上的平均准确度|
|set_params（**参数）| 设置此估算器的参数|
|sparsify（） | 将系数矩阵转换为稀疏格式|