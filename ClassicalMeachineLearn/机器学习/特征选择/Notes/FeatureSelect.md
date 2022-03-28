# 【机器学习】特征选择(Feature Selection)方法汇总

## 介绍

`特征选择`是 `特征工程`里的一个重要问题，其目标是 **寻找最优特征子集** 。特征选择能剔除不相关(irrelevant)或冗余(redundant )的特征，从而达到减少特征个数， **提高模型精确度，减少运行时间的目的** 。另一方面，选取出真正相关的特征简化模型，协助理解数据产生的过程。并且常能听到“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已”，由此可见其重要性。但是它几乎很少出现于机器学习书本里面的某一章。然而在机器学习方面的成功很大程度上在于如果使用特征工程。

之所以要考虑特征选择，是因为机器学习经常面临过拟合的问题。 **过拟合**的表现是模型参数 **太贴合训练集数据** ，模型在训练集上效果很好而在测试集上表现不好，也就是在高方差。简言之模型的泛化能力差。过拟合的原因是模型对于训练集数据来说太复杂，要解决过拟合问题，一般考虑如下方法：

1. 收集更多数据
2. 通过正则化引入对复杂度的惩罚
3. 选择更少参数的简单模型
4. 对数据降维（降维有两种方式：特征选择和特征抽取）

其中第1条一般是很难做到的，一般主要采用第2和第4点

## 一般流程

特征选择的一般过程：

1. 生成子集：搜索特征子集，为评价函数提供特征子集
2. 评价函数：评价特征子集的好坏
3. 停止准则：与评价函数相关，一般是阈值，评价函数达到一定标准后就可停止搜索
4. 验证过程：在验证数据集上验证选出来的特征子集的有效性

但是， 当特征数量很大的时候， 这个搜索空间会很大，如何找最优特征还是需要一些经验结论。

## 三大类方法

根据特征选择的形式，可分为三大类：

* Filter(过滤法)：按照 `发散性`或 `相关性`对各个特征进行评分，设定阈值或者待选择特征的个数进行筛选
* Wrapper(包装法)：根据目标函数（往往是预测效果评分），每次选择若干特征，或者排除若干特征
* Embedded(嵌入法)：先使用某些机器学习的模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征（类似于Filter，只不过系数是通过训练得来的）

## 过滤法

基本想法是：分别对每个特征 x_i ，计算 x_i 相对于类别标签 y 的信息量 S(i) ，得到 n 个结果。然后将 n 个 S(i) 按照从大到小排序，输出前 k 个特征。显然，这样复杂度大大降低。那么关键的问题就是使用什么样的方法来度量 S(i) ，我们的目标是选取与 y 关联最密切的一些 特征x_i 。

* Pearson相关系数
* 卡方验证
* 互信息和最大信息系数
* 距离相关系数
* 方差选择法

### Pearson相关系数

皮尔森相关系数是一种最简单的，能帮助理解特征和响应变量之间关系的方法，衡量的是变量之间的线性相关性，结果的取值区间为**[-1,1]** ， -1 表示完全的负相关(这个变量下降，那个就会上升)， +1 表示完全的正相关， 0 表示没有线性相关性。Pearson Correlation速度快、易于计算，经常在拿到数据(经过清洗和特征提取之后的)之后第一时间就执行。Scipy的[pearsonr](https://link.zhihu.com/?target=https%3A//docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html)方法能够同时计算相关系数和p-value

```python
import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise：", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise：", pearsonr(x, x + np.random.normal(0, 10, size)))


from sklearn.feature_selection import SelectKBest
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

Pearson相关系数的一个**明显缺陷**是，作为特征排序机制，他 **只对线性关系敏感** 。如果关系是非线性的，即便两个变量具有一一对应的关系，Pearson相关性也可能会接近 0 。

### 卡方验证

经典的卡方检验是检验**类别型变量**对**类别型变量**的相关性。假设自变量有N种取值，因变量有M种取值，考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量：

![[公式]](https://www.zhihu.com/equation?tex=%5Cchi%5E2%3D%5Csum%5Cfrac%7B%28A-E%29%5E2%7D%7BE%7D+)

不难发现，这个统计量的含义简而言之就是自变量对因变量的相关性。用sklearn中feature_selection库的SelectKBest类结合卡方检验来选择特征的代码如下：

```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target  #iris数据集

#选择K个最好的特征，返回选择特征后的数据
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
```

[sklearn.feature_selection](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/classes.html%23module-sklearn.feature_selection)模块中的类可以用于样本集中的特征选择/维数降低，以提高估计器的准确度分数或提高其在非常高维数据集上的性能

### 互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)

经典的互信息也是评价**类别型变量**对**类别型变量**的相关性的，互信息公式如下：

![[公式]](https://www.zhihu.com/equation?tex=MI%28x_i%2Cy%29%3D%5Csum_%7Bx_i%5Cin%7B0%2C1%7D%7D%5Csum_%7By%5Cin%7B0%2C1%7D%7Dp%28x_i%2Cy%29log%5Cfrac%7Bp%28x_i%2Cy%29%7D%7Bp%28x_i%29p%28y%29%7D+)

当x_i是0/1离散值的时候，这个公式如上。很容易推广到 x_i 是多个离散值的情况。这里的 p(x_i,y) , p(x_i) 和 p(y) 都是从训练集上得到的。若问这个 MI 公式如何得来，请看它的 KL 距离（Kullback-Leibler）表述： ![[公式]](https://www.zhihu.com/equation?tex=+MI%28x_i%2Cy%29%3DKL%28P%28x_i%2Cy%29%7C%7Cp%28x_i%29p%28y%29%29++) 也就是说, MI 衡量的是 x_i 和 y 的独立性。如果它俩独立 P(x_i,y)=p(x_i)p(y) ，那么 KL 距离值为0，也就是 x_i 和 y 不相关了，可以去除 x_i 。相反，如果两者密切相关，那么 MI 值会很大。在对 MI 进行排名后，最后剩余的问题就是如何选择 k 个值（前 k 个 x_i ）。(后面将会提到此方法)我们继续使用交叉验证的方法，将 k 从 1 扫描到 n ，取评分最高的k 。 不过这次复杂度是线性的了。比如，在使用朴素贝叶斯分类文本的时候，词表长度 n 很大。 使用filiter特征选择方法，能够增加分类器精度。

想把互信息直接用于特征选择其实不是太方便：

1. 它不属于度量方式，也没有办法归一化，在不同数据及上的结果无法做比较
2. 对于连续变量的计算不是很方便（ X 和 Y 都是集合, x_i, y 都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感

**最大信息系数**克服了这两个问题。它首先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在 [0,1] 。[minepy](https://link.zhihu.com/?target=https%3A//minepy.readthedocs.io/en/latest/)提供了MIC功能。

下面我们来看下 y=x^2 这个例子，MIC算出来的互信息值为1(最大的取值)。代码如下：

```python
from minepy import MINE

m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print(m.mic())


from sklearn.feature_selection import SelectKBest
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
# 选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

### 距离相关系数

距离相关系数是为了克服Pearson相关系数的弱点而生的。在 x 和 x^2 这个例子中，即便Pearson相关系数是 0 ，我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是 0 ，那么我们就可以说这两个变量是独立的。

R的[energy](https://link.zhihu.com/?target=https%3A//cran.r-project.org/web/packages/energy/index.html)包里提供了距离相关系数的实现，另外这是[Python gist](https://link.zhihu.com/?target=https%3A//gist.github.com/josef-pkt/2938402)的实现。

```text
# R-code
> x = runif (1000, -1, 1)
> dcor(x, x**2)
[1] 0.4943864
```

尽管有MIC和距离相关系数在了，但当变量之间的关系接近线性相关的时候，Pearson相关系数仍然是不可替代的。

第一、Pearson相关系数计算速度快，这在处理大规模数据的时候很重要。

第二、Pearson相关系数的取值区间是[-1，1]，而MIC和距离相关系数都是[0，1]。这个特点使得Pearson相关系数能够表征更丰富的关系，符号表示关系的正负，绝对值能够表示强度。当然，Pearson相关性有效的前提是两个变量的变化关系是单调的。

### 方差选择法

过滤特征选择法还有一种方法不需要度量特征 x_i 和类别标签 y 的信息量。这种方法先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。

例如，假设我们有一个具有布尔特征的数据集，并且我们要删除超过80％的样本中的一个或零（开或关）的所有特征。布尔特征是伯努利随机变量，这些变量的方差由下式给出: ![[公式]](https://www.zhihu.com/equation?tex=Var%5BX%5D%3Dp%281-p%29+)

[VarianceThreshold](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html%23sklearn.feature_selection.VarianceThreshold)是特征选择的简单基线方法。它删除方差不符合某个阈值的所有特征。默认情况下，它会删除所有零差异特征，即所有样本中具有相同值的特征。代码如下：

```python
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
print(sel.fit_transform(X))

# VarianceThreshold(threshold=3).fit_transform(iris.data)
```

输出结果：

array([[0, 1],  [1, 0],  [0, 0],  [1, 1],  [1, 0],  [1, 1]]) 如预期的那样，VarianceThreshold已经删除了第一列，其具有 p=5/6>0.8 包含零的概率。

方差选择的逻辑并不是很合理，这个是基于各特征分布较为接近的时候，才能以方差的逻辑来衡量信息量。但是如果是离散的或是仅集中在几个数值上，如果分布过于集中，其信息量则较小。而对于连续变量，由于阈值可以连续变化，所以信息量不随方差而变。 实际使用时，可以结合cross-validate进行检验

## 包装法

基本思想：基于hold-out方法，对于每一个待选的特征子集，都在训练集上训练一遍模型，然后在测试集上根据误差大小选择出特征子集。需要先选定特定算法，通常选用普遍效果较好的算法， 例如Random Forest， SVM， kNN等等。

> 西瓜书上说包装法应该欲训练什么算法，就选择该算法进行评估
> 随着学习器（评估器）的改变，最佳特征组合可能会改变

贪婪搜索算法（greedy search）是局部最优算法。与之对应的是穷举算法 (exhaustive search)，穷举算法是遍历所有可能的组合达到全局最优级，但是计算复杂度是2^n，一般是不太实际的算法。

### 前向搜索

前向搜索说白了就是，每次增量地从剩余未选中的特征选出一个加入特征集中，待达到阈值或者 n 时，从所有的 F 中选出错误率最小的。过程如下：

1. 初始化特征集 F 为空。
2. 扫描 i 从 1 到 n 如果第 i 个特征不在 F 中，那么特征 i 和F 放在一起作为 F_i (即 F_i=F\cup{i} )。 在只使用 F_i 中特征的情况下，利用交叉验证来得到 F_i 的错误率。
3. 从上步中得到的 n 个 F_i 中选出错误率最小的 F_i ,更新 F 为 F_i 。
4. 如果 F 中的特征数达到了 n 或者预定的阈值（如果有的话）， 那么输出整个搜索过程中最好的 ；若没达到，则转到 2，继续扫描。

### 后向搜索

既然有增量加，那么也会有增量减，后者称为后向搜索。先将 F 设置为 {1,2,...,n} ，然后每次删除一个特征，并评价，直到达到阈值或者为空，然后选择最佳的 F 。

这两种算法都可以工作，但是计算复杂度比较大。时间复杂度为：![[公式]](https://www.zhihu.com/equation?tex=O%28n%2B%28n-1%29%2B%28n-2%29%2B...%2B1%29%3DO%28n%5E2%29+)

### 递归特征消除法

递归消除特征法使用一个 `基模型`来进行多轮训练，每轮训练后通过学习器返回的 coef_ 或者feature_importances_ 消除若干权重较低的特征，再基于新的特征集进行下一轮训练。

使用feature_selection库的RFE类来选择特征的代码如下：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```

## 嵌入法

* 基于惩罚项的特征选择法 通过L1正则项来选择特征：L1正则方法具有稀疏解的特性，因此天然具备特征选择的特性。

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

#带L1惩罚项的逻辑回归作为基模型的特征选择   
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
```

要注意，L1没有选到的特征不代表不重要，原因是两个具有高相关性的特征可能只保留了一个，如果要确定哪个特征重要应再通过L2正则方法交叉检验。

```python
from sklearn.feature_selection import SelectFromModel

#带L1和L2惩罚项的逻辑回归作为基模型的特征选择   
#参数threshold为权值系数之差的阈值   
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
```

* 基于学习模型的特征排序 这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。通过这种训练对特征进行打分获得相关性后再训练最终模型。

在[波士顿房价数据集](https://link.zhihu.com/?target=https%3A//archive.ics.uci.edu/ml/datasets/Housing)上使用sklearn的[随机森林回归](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)给出一个**单变量选择**的例子：

```python
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

#加载波士顿房价作为数据集
boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

#n_estimators为森林中树木数量，max_depth树的最大深度
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
    #每次选择一个特征，进行交叉验证，训练集和测试集为7:3的比例进行分配，
    #ShuffleSplit()函数用于随机抽样（数据集总数，迭代次数，test所占比例）
    score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                               cv=ShuffleSplit(len(X), 3, .3))
    scores.append((round(np.mean(score), 3), names[i]))

#打印出各个特征所对应的得分
print(sorted(scores, reverse=True))
```

输出结果：

[(0.64300000000000002, 'LSTAT'), (0.625, 'RM'), (0.46200000000000002, 'NOX'), (0.373, 'INDUS'), (0.30299999999999999, 'TAX'), (0.29799999999999999, 'PTRATIO'), (0.20399999999999999, 'RAD'), (0.159, 'CRIM'), (0.14499999999999999, 'AGE'), (0.097000000000000003, 'B'), (0.079000000000000001, 'ZN'), (0.019, 'CHAS'), (0.017999999999999999, 'DIS')]

发布于 2019-07-18 17:36

[机器学习](https://www.zhihu.com/topic/19559450)

[特征工程](https://www.zhihu.com/topic/20058170)

[特征选择](https://www.zhihu.com/topic/19809410)

赞同 43423 条评论
