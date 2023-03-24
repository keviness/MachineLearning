> 🔗 原文链接： [https://blog.csdn.net/qq_24591139/a...](https://blog.csdn.net/qq_24591139/article/details/100085359)

> ⏰ 剪存时间：2023-03-15 13:07:53 (UTC+8)

> ✂️ 本文档由 [飞书剪存 ](https://www.feishu.cn/hc/zh-CN/articles/606278856233?from=in_ccm_clip_doc)一键生成

# Lightgbm原理、参数详解及python实例

# 预备知识：GDBT

1)对所有特征都按照特征的数值进行预排序。
2)在遍历分割点的时候用O(#data)的代价找到一个特征上的最好分割点。
3)找到一个特征的分割点后，将数据分裂成左右子节点。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=OGVlNDkwNzdjMGYwODRmMzQ2MzNiZTlkNjFkOGEyZmFfNld0d0c1M05WaUZXTUxCMG80N3JqUWhpeU53VldPYzlfVG9rZW46RHRYQmJkRmR1b3lMbkV4UzRpSWMyQ2NubmVmXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

# [LightGBM](https://so.csdn.net/so/search?q=LightGBM&spm=1001.2101.3001.7020)

LightGBM是个快速的，分布式的，高性能的基于决策树算法的梯度提升框架。可用于排序，分类，回归以及很多其他的机器学习任务中。
●训练时样本点的采样优化：保留梯度较大的样本
●特征维度的优化：互斥特征绑定与合并
●决策树生成：特征分割，生长策略
●直接处理类别特征：统计类别数量

优点：
1、 更快的训练速度和更高的效率：GOSS算法，EFB算法、基于直方图的算法；
2、 降低内存使用：使用离散的箱子(bins)保存并替换连续值
3、 精度更高：leaf-wise分裂方法，同时使用max-depth 参数防止过拟合
4、 支持并行和GPU学习
5、 能够处理大规模数据

# **使用GOSS算法和EFB算法的梯度提升树（GBDT）称之为LightGBM。**

# **在更高的处理效率和较低的内存前提下，不降低精度**

## 一、原理

### 1.单边梯度采样算法（Grandient-based One-Side Sampling，GOSS）

核心作用：训练集样本采样优化
1）保留梯度较大的样本；
2） 对梯度较小的样本进行随机抽样；
3）在计算增益时，对梯度较小的样本增加权重系数.

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NjNhYWRmOWUxYjE0MmNlNGE0NTNlYTQ2OTA1NGM4ZGVfdlZGZUpZcnZEVVdsUmlqOWdnSExGZTRlMVUwSG9rSGJfVG9rZW46VDVidmJ0eHZMb3JsVVB4Wm5rTGNRN2l1bmZoXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

算法描述：
输入：训练数据，迭代步数d，大梯度数据的采样率a，小梯度数据的采样率b，损失函数和若学习器的类型（一般为决策树）；

输出：训练好的强学习器；

（1）根据样本点的梯度的绝对值进行降序排序；

（2）对排序后的结果选取前a*100%的样本生成一个大梯度样本点的子集；

（3）对剩下的样本集合（1-a）  *100%的样本，随机的选取b * （1-a）*100%个样本点，生成一个小梯度样本点的集合；

（4）将大梯度样本和采样的小梯度样本合并；

（5）使用上述的采样的样本，学习一个新的弱学习器；

（6）在新的弱学习器中，计算增益时将小梯度样本乘上一个权重系数（1-a)/b；

（7）不断地重复（1）~（6）步骤直到达到规定的迭代次数或者收敛为止。

### 2.Exclusive Feature Bundling 算法(EFB)

核心作用：特征抽取，将互斥特征（一个特征值为零,一个特征值不为零）绑定在一起，从而减少特征维度。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MTc3N2UxOWI2OTAwOGRkOWZiOGZmZDk0ODYxYzdjZjZfSVV1QVUxaXlLR3EzemxmakVMUncxeDU4dWFiVHlqVGxfVG9rZW46SEs2VWJJbDNsb0lDVk14eTNiMGNoWGtGblFoXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=Nzk1MjI3NzQwYWVmOWJjYmM3Y2QzOTUwYzgwOWJhOTNfZkFuYWN4Qlk1R1F0SU9zcklDdnFQTW5LZndoSHRrbFpfVG9rZW46UlhwRWI3ZmFsb0Z4WUd4UUsya2NzU0xablFmXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

算法3：确定哪些特征被捆绑；
算法4：怎样合并被捆绑特征

算法3描述：
输入：特征F，最大冲突数K，图G；
输出：特征捆绑集合bundles；

（1）构造一个边带有权重的图，其权值对应于特征之间的总冲突；

（2）通过特征在图中的度来降序排序特征；

（3）检查有序列表中的每个特征，并将其分配给具有小冲突的现有bundling（由控制），或创建新bundling。

###### 更高效EBF的算法步骤如下：

1）将特征按照非零值的个数进行排序
2）计算不同特征之间的冲突比率
3）遍历每个特征并尝试合并特征（Histogram算法），使冲突比率最小化

参考文献：https://blog.csdn.net/qq_24519677/article/details/82811215

### 3.直方图算法（Histogram算法）

#### 3.1 核心思想：

将连续的特征值离散化成K个整数（bin数据），构造宽度为K的直方图，遍历训练数据，统计每个离散值在直方图中的累积统计量。在选取特征的分裂点的时候，只需要遍历排序直方图的离散值。

● 使用bin替代原始数据相当于增加了正则化；
● 使用bin很多数据的细节特征被放弃，相似的数据可能被划分到一起，数据之间的差异消失；
● bin数量的选择决定了正则化的程度，K越少惩罚越严重，欠拟合风险越高

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGFkYjJmNmVmOGY4OGEyZmRjZTMxMmI5MDgyZGVmMzBfT1ZsT0FmbDdTNkpXaVl0a3VWUzJQY0dkVjJYS2doV1hfVG9rZW46Uk9DcGJKVWllb1Z5UXR4RzRHNmNENGJpbnVoXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

#### 3.2 直方图加速

一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=YjFkNjljMWVkYzM5YTE3ZjMzNGYxNzg5MjBjZjUzMjZfYjE3ODdVd0VHNUFXWDFhQVNpb2tqM21VNlhLckFnSUlfVG9rZW46RU9MYWI4dDRLb091Tk14d1A0eWNQSjBsbk9LXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

#### 3.3 合并被绑定特征

将bundle内不同的特征加上一个偏移常量，使不同特征的值分布到bundle的不同bin内。例如：特征A的取值范围为[0,10)，特征B的原始取值范围为[0，20)，对特征B的取值上加一个偏置常量10，将其取值范围变为[10,30)，这样就可以将特征A和B绑定在一起了。

### 4、决策树生长策略

level_wise:多线程优化，控制模型复杂度，不易过拟合。
leaf-wise：计算代价较小，更精确，易过拟合（map_depth）。

LightGBM采用leaf-wise生长策略，每次从当前所有叶子中找到分裂增益最大（一般也是数据量最大）的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个map_depth的限制，在保证高效率的同时防止过拟合。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2ZiMWYxZmI5ZjdjNDJjOTZlMDI3ZGRmM2JkNWI3YmRfZHJUTWNDMk9jQ1V3RU0waklEVnBDTjlPTGdHNjFVeVpfVG9rZW46UVpxbGJjQ3E2b3ByTWt4RU1Ec2NrT3dvbkRlXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

### 5.直接处理类别特征

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MTJjM2YwZTVkOWE5ZTk0NzE0MWY1ZjMyYjk5ZjFhMTBfWDI5MWU3SnAxNEtvcUlDQWpBTjRvVVBJWDVYeDRTRlpfVG9rZW46QW9DNWIyQmdwb0kzdmN4alhiVWM1WXFsbk9mXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

##### 详细流程：

1、离散特征建立直方图的过程

统计该特征下每一种**离散值出现的次数，并从高到低排序，**并过滤掉出现次数较少的特征值, 然后为每一个特征值，建立一个bin容器, 对于在bin容器内出现次数较少的特征值直接过滤掉，不建立bin容器。

2、计算分裂阈值的过程：

2.1 先看该特征下划分出的bin容器的个数，如果bin容器的数量小于4，直接使用one vs other方式, 逐个扫描每一个bin容器，找出最佳分裂点;

2.2 对于bin容器较多的情况, 先进行过滤，只让子集合较大的bin容器参加划分阈值计算, 对每一个符合条件的bin容器进行公式计算(公式如下: 该bin容器下所有样本的一阶梯度之和 / 该bin容器下所有样本的二阶梯度之和 + 正则项(参数cat_smooth)，这里为什么不是label的均值呢？其实上例中只是为了便于理解，只针对了学习一棵树且是回归问题的情况， 这时候一阶导数是Y, 二阶导数是1)，得到一个值，根据该值对bin容器从小到大进行排序，然后分从左到右、从右到左进行搜索，得到最优分裂阈值。但是有一点，没有搜索所有的bin容器，而是设定了一个搜索bin容器数量的上限值，程序中设定是32，即参数max_num_cat。
LightGBM中对离散特征实行的是many vs many 策略，这32个bin中最优划分的阈值的左边或者右边所有的bin容器就是一个many集合，而其他的bin容器就是另一个many集合。

2.3 对于连续特征，划分阈值只有一个，对于离散值可能会有多个划分阈值，每一个划分阈值对应着一个bin容器编号，当使用离散特征进行分裂时，只要数据样本对应的bin容器编号在这些阈值对应的bin集合之中，这条数据就加入分裂后的左子树，否则加入分裂后的右子树。

### 6、并行学习

LightGBM原生支持并行学习，目前支持特征并行(Featrue Parallelization)和数据并行(Data Parallelization)两种，还有一种是基于投票的数据并行(Voting Parallelization)。
●特征并行的主要思想是在不同机器、在不同的特征集合上分别寻找最优的分割点，然后在机器间同步最优的分割点。
●数据并行则是让不同的机器先在本地构造直方图，然后进行全局的合并，最后在合并的直方图上面寻找最优分割点。

LightGBM针对这两种并行方法都做了优化。
●特征并行算法中，通过在本地保存全部数据避免对数据切分结果的通信。
●数据并行中使用分散规约 (Reduce scatter) 把直方图合并的任务分摊到不同的机器，降低通信和计算，并利用直方图做差，进一步减少了一半的通信量。
●基于投票的数据并行(Voting Parallelization)则进一步优化数据并行中的通信代价，使通信代价变成常数级别。在数据量很大的时候，使用投票并行可以得到非常好的加速效果。

使用场景：
特征并行：数据量小，但特征数量多
数据并行：数据量较大，特征数量少
投票并行：数据量大，特征数量多
————————————————

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2YwY2EzNmJmNThhMjUwZWMyNzg5OTYwMTg5NGE1ZDZfU2gwWkNuZmxUbHM5am1lZnNOakxoNlpaNnlnR3lwUDlfVG9rZW46Wm85OWJXbk5jb2Z5S0V4eFFpZmMxd1hkbjViXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=M2IxZmU3MDJhOTlmNjFjMTIzODUwOGRlZjA2ZjllM2RfbWJWWnZXRlloS3pCdTB6dDlVazExRll0RUhISHBPbUlfVG9rZW46Q2kxeGJyaWhhbzVyeFd4Z2s2Z2NhZHc3blpiXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkwODE2NzI2ZDU2NGVlYWJhZGIwOThhNzM1OWI3MzhfazZiQXlJR2taNU9JSXpNSzRhQUhOSzBvVVB0S2R2aWhfVG9rZW46Uk9uVWIyY0Jxb2Ruazd4NGdzVGNHclV1bmNlXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

### 7、存储

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=ODM4ZWJiYmJmNDI2YmU5OGJiNGMyNDliMmEwMWViNjhfRmRDeVRCRzk3SXAwY2pjRnhDRDZOZXo2UGIxYWI4c3RfVG9rZW46TXBhWGJ0R2ZXbzBHRzd4dmV4cmN4N2JHbmFmXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

### 8、其他

●当生长相同的叶子时，Leaf-wise 比 level-wise 减少更多的损失。
●高速，高效处理大数据，运行时需要更低的内存，支持 GPU
●不要在少量数据上使用，会过拟合，建议 10,000+ 行记录时使用。

#### 9. XGBoost与LightGBM对比

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2FlZjI1ZGI3OWE4MjdjZmU0MDIwNDE4MjExZGU4YTBfUkVtSWw5MlZrUGlnaFB0YWRrZGlXaThlRjhvdWhvc2NfVG9rZW46WUV5cmJQaXExb05JajN4TEU3SmMzREhDbjBlXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

# 二、使用

## 1、安装

```Plaintext
pip install setuptools wheel numpy scipy scikit-learn -U
pip install lightgbm
```

## 2、数据接口

LightGBM Python 模块能够使用以下几种方式来加载数据:
●libsvm/tsv/csv txt format file（libsvm/tsv/csv 文本文件格式）
●Numpy 2D array, pandas object（Numpy 2维数组, pandas 对象）
●LightGBM binary file（LightGBM 二进制文件）

加载后的数据存在 Dataset 对象中.

```Plaintext
train_data = lgb.Dataset('train.svm.bin')
```

要加载 numpy 数组到 Dataset 中:

```Plaintext
data = np.random.rand(500, 10)  # 500 个样本, 每一个包含 10 个特征
label = np.random.randint(2, size=500)  # 二元目标变量,  0 和 1
train_data = lgb.Dataset(data, label=label)
```

要加载 scpiy.sparse.csr_matrix 数组到 Dataset 中:

```Plaintext
csr = scipy.sparse.csr_matrix((dat, (row, col)))
train_data = lgb.Dataset(csr)
```

保存 Dataset 到 LightGBM 二进制文件将会使得加载更快速:

```Plaintext
train_data = lgb.Dataset('train.svm.txt')
train_data.save_binary('train.bin')
```

创建验证数据:

```Plaintext
1）test_data = train_data.create_valid('test.svm')
```

```Plaintext
2）test_data = lgb.Dataset('test.svm', reference=train_data)
```

## 3.设置参数

```Plaintext
# 将参数写成字典形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数     ####regression默认regression_l2
    'metric': {'l2', 'auc'},  # 评估函数
    'max_depth': 6     ###   树的深度           ###按层
    'num_leaves': 50  ###   由于leaves_wise生长，小于2^max_depth   #####按leaf_wise
    'learning_rate': 0.05,  # 学习速率
    'subsample'/'bagging_fraction':0.8           ###  数据采样
    'colsample_bytree'/'feature_fraction': 0.8  ###  特征采样
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
```

## 4.参数详解

pythonAPI官方文档：https://lightgbm.readthedocs.io/en/latest/Python-API.html
参考文献：
https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst
https://blog.csdn.net/u012735708/article/details/83749703
https://www.jianshu.com/p/1100e333fcab

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NGZlOWI1YTFhNGM4NGIxMjM3ZWZkYmZkNmVjZDUyNzJfT1V0MTZNZVpNenJocXVvemdTUDZkWEZsTGdJaWR0TWlfVG9rZW46RHBlY2JTNFFTb0xseGR4dnc0UmNmU1pCbkJoXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NDU4MjU4ZTJjOTZlMmIzNDAwZjBkMTQwZDQzYTFhZTZfNDJOUFlRZ3BxUnZmZmVSa042QVhuRUhZOTJXdHpoejNfVG9rZW46UVJsSmIyb1BtbzRoOXN4QlVYSWNETUtmbkNjXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MjNiMzgyYWEzYTM2Njc0NjUzYWJhNTBiNmEyYTMxN2ZfYUE5REp0NDlwODZ4dnY0VFRrNlVDYjRIeEpzWWxsMm5fVG9rZW46TVVXUmJUbzRNb210aGd4N3l2SWNJTGpIbmhkXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MDE1YTAwOGY1ZmQ5Nzk4OWZiNmUzZjU4OWUwMzc5YjRfZTZ5dlNjUkpKZHVyS2NvZU9oWWdJc3E3djNlQ0piTlBfVG9rZW46WkF1eGJVa21yb2ZLbW94R2tWN2N4eElabmxvXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MGNiZjY3YmQ2NWFhYTRhOGZkNWM2NmI5OWY4YjYyNWFfa0ZUaDg2YllteDQwY2xtSnF4Z0tkQm1xMUJ5VWlPYWpfVG9rZW46TUdmd2JuTHFkb1RyNHZ4ZUtLRWNFTUJMbklmXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=Mjk5ZTVhM2I2MGZiNDY4YzFkNDkwMzVmMDdkMjY5ZGFfRlBDUTNRSW5GNDBsZHpDWXd3cEpoYTQybGE1SnhpZ1lfVG9rZW46TmNUWWJGQUVYb0hPaWF4ZjRmQ2NtM3ZIblFiXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

##### 调参方法

调参：https://www.imooc.com/article/43784?block_id=tuijian_wz
LightGBM的调参过程和RF、GBDT等类似，其基本流程如下：
●首先选择较高的学习率，大概0.1附近，这样是为了加快收敛的速度。这对于调参是很有必要的。
●对决策树基本参数调参：
1）max_depth和num_leaves
2）min_data_in_leaf和min_sum_hessian_in_leaf
3）feature_fraction和bagging_fraction
●正则化参数调参
●最后降低学习率，这里是为了最后提高准确率

（1）

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=YTNlMDcyZmNkODIyOGJjN2Q3OTY5MGEzMzM0MWQwMDNfN2pWd1pvcTFud1MzMWdzeW5vZFBDNXBpUUoyRkxZZTBfVG9rZW46SEpmV2JYS0l0b1l3M3F4TFVwOGNmdHdsbnNtXzE2Nzg4NTc1NzM6MTY3ODg2MTE3M19WNA)

（2）GridSearchCV调参

## 5.原生实例

```Plaintext
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
 
# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
 
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}
 
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
 
# 保存模型到文件
gbm.save_model('model.txt')
 
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
 
# 评估模型
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
```

#### 6.sklearn接口实例

```Plaintext
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
 
# 加载数据
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
 
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)
 
# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
 
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
 
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
 
# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
```

#### 原生API与sklearnAPI接口区别总结

我们需要注意以下几点：

1. 多分类时lgb.train除了’objective’:‘multiclass’,还要指定"num_class":5，而sklearn接口只需要指定’objective’:‘multiclass’。
2. lgb.train中正则化参数为"lambda_l1", “lambda_l1”，sklearn中则为’reg_alpha’, ‘reg_lambda’。
3. 迭代次数在sklearn中是’n_estimators’:300，在初始化模型时指定。而在lgb.train中则可在参数params中指定，也可在函数形参中指出。
