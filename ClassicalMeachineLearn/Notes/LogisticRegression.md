## 逻辑回归（Logistic Regression）

## 一，Logistic回归算法思想

* Logistic回归是一种二分类算法那，它利用的是Sigmoid函数阈值在[0,1]这个特性。
* Logistic回归进行分类的主要思想是：根据现有数据对分类边界线建立回归公式，以此进行分类。
* Logistic本质上是一个基于条件概率的判别模型(Discriminative Model)。

### （一）Sigmoid函数

![Logistic Regression 2](./imgs/LogisticRegression2.png)
![Logistic Regression 1](./imgs/LogisticRegression1.jpeg)

* python实现Sigmoid函数

```py
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
```

### （二）代价函数

* 代价函数，是对于一个样本而言的。给定一个样本，我们就可以通过这个代价函数求出，样本所属类别的概率，而这个概率越大越好，所以也就是求解这个代价函数的最大值。
  ![Cost Function1](./imgs/CostFunction1.png)
* 满足J(θ)的最大的θ值即是我们需要求解的模型。
  ![Cost Function2](./imgs/CostFunction2.png)

### （三）梯度上升算法

![梯度上升](./imgs/梯度上升.png)

* python实现梯度上升算法

```py
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
```

* 2，python实现随机梯度算法

```py
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)				#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   									#参数初始化										#存储每次更新的回归系数
	for j in range(numIter):										
		dataIndex = list(range(m))
		for i in range(m):		
			alpha = 4/(1.0+j+i)+0.01   	 			#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))		#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 															#返回
```

## 二，Logistic回归的一般过程：

* 收集数据：采用任意方法收集数据。
* 准备数据：由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
* 分析数据：采用任意方法对数据进行分析。
* 训练算法：大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
* 测试算法：一旦训练步骤完成，分类将会很快。
* 使用算法：
  1，我们需要输入一些数据，并将其转换成对应的结构化数值；
  2，基于训练好的回归系数，就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；
  3，在这之后，就可以在输出的类别上做一些其他分析工作。
* 其他：Logistic回归的目的是寻找一个非线性函数Sigmoid的最佳拟合参数，求解过程可以由最优化算法完成。

## 三，python实现Logistic回归

```py
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
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    #使用改进的随即上升梯度训练
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
```

## 四，sklearn module实现Logistic Regression

### （一）定义Logistic Regression

```py
sklearn.linear_model.LogisticRegression(
    penalty='l2', *,  # 用于指定惩罚中使用的规范。'newton-cg'，'sag'和'lbfgs'求解器仅支持l2罚分。仅“saga”求解器支持，“elasticnet”。如果为“none”（liblinear求解器不支持），则不应用任何正则化。
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
    ##参数详解：
    1，penalty：惩罚项，str类型，可选参数为l1和l2，默认为l2。用于指定惩罚项中使用的规范。newton-cg、sag和lbfgs求解算法只支持L2规范。L1G规范假设的是模型的参数满足拉普拉斯分布，L2假设的模型参数满足高斯分布，所谓的范式就是加上对参数的约束，使得模型更不会过拟合(overfit)，但是如果要说是不是加了约束就会好，这个没有人能回答，只能说，加约束的情况下，理论上应该可以获得泛化能力更强的结果。
    2，dual：对偶或原始方法，bool类型，默认为False。对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False。
    3，tol：停止求解的标准，float类型，默认为1e-4。就是求解到多少的时候，停止，认为已经求出最优解。
    c：正则化系数λ的倒数，float类型，默认为1.0。必须是正浮点型数。像SVM一样，越小的数值表示越强的正则化。
    4，fit_intercept：是否存在截距或偏差，bool类型，默认为True。
    intercept_scaling：仅在正则化项为"liblinear"，且fit_intercept设置为True时有用。float类型，默认为1。
    5，class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者'balanced'字符串，默认为不输入，也就是不考虑权重，即为None。如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。举个例子，比如对于0,1的二元模型，我们可以定义class_weight={0:0.9,1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。如果class_weight选择balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当class_weight为balanced时，类权重计算方法如下：n_samples / (n_classes * np.bincount(y))。n_samples为样本数，n_classes为类别数量，np.bincount(y)会输出每个类的样本数，例如y=[1,0,0,1,1],则np.bincount(y)=[2,3]。
    那么class_weight有什么作用呢？
    在分类模型中，我们经常会遇到两类问题：
    （1）第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。这时，我们可以适当提高非法用户的权重。
    （2）第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。
    6，random_state：随机数种子，int类型，可选参数，默认为无，仅在正则化优化算法为sag,liblinear时有用。
    7，solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：
    （1）liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
    （2）lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    （3）newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
    （4）sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
    （5）saga：线性收敛的随机优化算法的的变重。
    8，总结：
    liblinear适用于小数据集，而sag和saga适用于大数据集因为速度更快。
    对于多分类问题，只有newton-cg,sag,saga和lbfgs能够处理多项损失，而liblinear受限于一对剩余(OvR)。啥意思，就是用liblinear的时候，如果是多分类问题，得先把一种类别作为一个类别，剩余的所有类别作为另外一个类别。一次类推，遍历所有类别，进行分类。
    newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear和saga通吃L1正则化和L2正则化。
    同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。
    从上面的描述，大家可能觉得，既然newton-cg, lbfgs和sag这么多限制，如果不是大样本，我们选择liblinear不就行了嘛！错，因为liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
    9，max_iter：算法收敛最大迭代次数，int类型，默认为10。仅在正则化优化算法为newton-cg, sag和lbfgs才有用，算法收敛的最大迭代次数。
    10，multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。
    11，OvR和MvM有什么不同？
    （1）OvR的思想很简单，无论你是多少元逻辑回归，我们都可以看做二元逻辑回归。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类样本以外的所有样本都作为负例，然后在上面做二元逻辑回归，得到第K类的分类模型。其他类的分类模型获得以此类推。
    （2）而MvM则相对复杂，这里举MvM的特例one-vs-one(OvO)作讲解。如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。
    （3）可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而MvM分类相对精确，但是分类速度没有OvR快。如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。
    12，verbose：日志冗长度，int类型。默认为0。就是不输出训练过程，1的时候偶尔输出结果，大于1，对于每个子模型都输出。
    13，warm_start：热启动参数，bool类型。默认为False。如果为True，则下一次训练是以追加树的形式进行（重新使用上一次的调用作为初始化）。
    14，n_jobs：并行数。int类型，默认为1。1的时候，用CPU的一个内核运行程序，2的时候，用CPU的2个内核运行程序。为-1的时候，用所有CPU的内核运行程序。
```

### （三）示例

```py
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
```

### （四）方法

|             Method             |             Description             |
| :-----------------------------: | :----------------------------------: |
|     decision_function（X）     |         预测样本的置信度得分         |
|           densify（）           |     将系数矩阵转换为密集数组格式     |
|  fit（X，y [，sample_weight]）  |      根据给定的训练数据拟合模型      |
|      get_params（[深的]）      |          获取此估计量的参数          |
|          predict（X）          |        预测X中样本的类别标签        |
|     predict_log_proba（X）     |          预测概率估计的对数          |
|       predict_proba（X）       |               概率估计               |
| score（X，y [，sample_weight]） | 返回给定测试数据和标签上的平均准确度 |
|      set_params（**参数）      |          设置此估算器的参数          |
|          sparsify（）          |       将系数矩阵转换为稀疏格式       |
