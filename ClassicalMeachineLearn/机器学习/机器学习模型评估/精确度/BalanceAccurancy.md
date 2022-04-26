# sklearn中分类模型评估指标（一）：准确率、Top准确率、平衡准确率

准确率分数

`accuracy_score`函数计算准确率分数，即预测正确的分数（默认）或计数（当normalize=False时）。

在多标签分类中，该函数返回子集准确率（subset accuracy）。 如果样本的整个预测标签集与真实标签集严格匹配，则子集准确率为 1.0； 否则为 0.0。

如果\hat{y}_i**y**^i是第i个样本的预测值和y_i**y**i是对应的真实值，那么正确预测的分数，公式定义如下：

\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)=\frac{TP+TN}{TP+FP+TN+FN}**accuracy**(**y**,**y**^****)**=**n**samples****1****i**=**0**∑**n**samples****−**1****1**(**y**^****i****=**y**i****)**=**T**P**+**F**P**+**T**N**+**F**N**T**P**+**T**N**

其中，1(x)**1**(**x**) 表示指示函数（indicator function），它的含义是：当输入为True的时候，输出为1，输入为False的时候，输出为0。

> **关于指示函数的说明：**
>
> 在数学中，**指示函数**是定义在某集合X**X**上的函数，表示其中有哪些元素属于某一子集A**A** ，常应用在集合论中。指示函数有时候也称为特征函数。

示例代码如下：

```python
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

print(accuracy_score(y_true, y_pred))
print(accuracy_score(y_true, y_pred, normalize=False))
复制代码
```

运行结果：

```
0.5
2
复制代码
```

在具有两个类标签指示符矩阵的多标签场景下，示例代码为：

```python
print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
复制代码
```

运行结果：

```
0.5
复制代码
```

## Top-k准确率分数

`top_k_accuracy_score`函数是对 `accuracy_score`函数的扩展。 不同之处在于，只要真实标签与前 k 个最高预测分数之一相关联，就认为预测是正确的。`accuracy_score`是 k = 1的特例。

该函数可以应用于二分类和多分类情况，但不包括多标签情况。

如果\hat{f}_{i,j}**f**^i**,**j****是对应于第i个样本的第j个最大预测分数的预测类别，y_i**y**i****是对应的真实值，那么对于n_\text{samples}**n**samples个样本，正确预测的分数被定义为

\texttt{top-k accuracy}(y, \hat{f}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} \sum_{j=1}^{k} 1(\hat{f}_{i,j} = y_i)**top-k accuracy**(**y**,**f**^****)**=**n**samples****1****i**=**0**∑**n**samples****−**1****j**=**1**∑**k****1**(**f**^****i**,**j****=**y**i****)

其中，k是允许的预测个数， 1(x)**1**(**x**)是指示函数。

示例代码：

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score
y_true = np.array([0, 1, 2, 2])

# 0, 1, 2
y_score = np.array([[0.5, 0.2, 0.2], # 0,1
                    [0.3, 0.4, 0.2], # 0,1
                    [0.2, 0.4, 0.3], # 1,2
                    [0.7, 0.2, 0.1]]) # 0,1

print(top_k_accuracy_score(y_true, y_score, k=2))

# 如果没有归一化，则返回分类样本预测正确的数量
print(top_k_accuracy_score(y_true, y_score, k=2, normalize=False))
复制代码
```

**运行结果：**

```
0.75
3
复制代码
```

## 平衡准确率分数

`balance_accuracy_score` 函数计算平衡准确率，在二分类和多分类场景中，平衡准确率用来处理不平衡数据集的问题，从而避免对不平衡数据集的评估表现夸大。它被定义为在每个类上的召回率的宏平均值，或者等效于，原始准确率（raw accuracy），其中每个样本根据其真实类别的逆流行程度（逆流行率）进行加权。 因此，对于平衡数据集，其分数等于准确率分数。

在二分类情况下，平衡准确率等于灵敏度（true positive rate，真阳性率）和特异度（true negative rate，真阴性率）的算术平均值，或者二分类情况下，预测的 ROC 曲线下面积而不是分数：

\texttt{balanced-accuracy} = \frac{1}{2}\left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right )=\frac{TPR+TNR}{2}**balanced-accuracy**=**2**1****(**T**P**+**F**N**T**P****+**T**N**+**F**P**T**N****)**=**2**T**P**R**+**T**N**R**

如果分类器在任一类上具有同等表现，则该术语简化为常规准确率（即正确预测的数量除以预测总数）。

相反，如果仅因为分类器使用了不平衡的测试集，导致常规的准确率高于随机值（chance=\frac{1}{n\_classes}**c**h**a**n**c**e**=**n**_**c**l**a**s**s**e**s**1**），那么平衡准确率，这种情况下，将下降到 \frac{1}{n\_classes}**n**_**c**l**a**s**s**e**s**1。

当 `adjusted=False`时，分数范围为从0到1，最佳值为1，最差值为0。当 `adjusted=True` 时，分数范围从\frac{1}{1 - n\_classes}**1**−**n**_**c**l**a**s**s**e**s**1到1**1**（包括边界），随机值分数表现是0（不平衡数据集），完美表现分数是1，完全预测错误分数为\frac{1}{1 - n\_classes}**1**−**n**_**c**l**a**s**s**e**s**1。

如果 y_i**y**i是第i**i**个样本的真实值，并且w_i**w**i是对应的样本权重，那么我们调整样本权重为：

\hat{w}_i = \frac{w_i}{\sum_j{1(y_j = y_i) w_j}}**w**^**i****=**∑**j****1**(**y**j=**y**i****)**w**j****w**i******

其中，1(x)**1**(**x**)是指示函数。给定样本 i**i**的预测\hat{y}_i**y**^i，则平衡准确率公式定义为：

\texttt{balanced-accuracy}(y, \hat{y}, w) = \frac{1}{\sum{\hat{w}_i}} \sum_i 1(\hat{y}_i = y_i) \hat{w}_i**balanced-accuracy**(**y**,**y**^,**w**)**=**∑**w**^**i****1****i**∑1**(**y**^i=**y**i)**w**^**i****

adjusted相关源码如下：

```python
if adjusted:
    n_classes = len(per_class) # 类别数
    chance = 1 / n_classes
    score -= chance
    score /= 1 - chance
复制代码
```

针对二分类情况，示例代码：

```python
from sklearn.metrics import balanced_accuracy_score
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]
# tp=1,  fn=1,  tn=3, fp=1
# 常规：(1+3)/6 = 0.66
# 平衡：(1/2+3/4)/2 = 0.625
print(accuracy_score(y_true, y_pred))
print(balanced_accuracy_score(y_true, y_pred))
# 1/类别数
# 0.625 - 1/2  = 0.125
# 0.125 / (1-1/2) = 0.25
print(balanced_accuracy_score(y_true, y_pred, adjusted=True))
复制代码
```

运行结果：

```
0.6666666666666666
0.625
0.25
复制代码
```

针对多分类情况，示例代码如下：

```python
from sklearn.metrics import accuracy_score,balanced_accuracy_score

y_true = [0, 1, 2, 0, 0]
y_pred = [0, 2, 2, 0, 1]
# 3/5
print(accuracy_score(y_true, y_pred))
# 对于0  tp=2  fn=1     2/3
# 对于1  tp=0  fn=1     0
# 对于2  tp=1  fn=0     1
# (2/3+0+1)/3 = 5/9
print(balanced_accuracy_score(y_true, y_pred, adjusted=False))
# 5/9 - 1/3 = 2/9
# (2/9)/(1-1/3) = 1/3
print(balanced_accuracy_score(y_true, y_pred, adjusted=True))
复制代码
```

运行结果：

```
0.6
0.5555555555555555
0.3333333333333332
复制代码
```

针对不平衡数据集的示例代码如下：

```python
from sklearn.metrics import recall_score,balanced_accuracy_score 

def test_balanced_accuracy_score():
    y_true = [0, 1, 2, 0, 0, 1, 4]
    y_pred = [0, 2, 2, 0, 1, 1, 2]
    macro_recall = recall_score(y_true, y_pred, average='macro',
                                labels=np.unique(y_true))
    # adjusted=False时，平衡准确率
    balanced = balanced_accuracy_score(y_true, y_pred)
    print(balanced)
    # adjusted=True时，平衡准确率
    adjusted = balanced_accuracy_score(y_true, y_pred, adjusted=True)
    print(adjusted)
    print("-------------")
    # 不平衡数据集（预测的运气值）
    print(np.full_like(y_true, y_true[0]))
    print(np.full_like(y_true, y_true[1]))
    print(np.full_like(y_true, y_true[2]))
    print(np.full_like(y_true, y_true[6]))

    # 不平衡数据集（预测的运气值）
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[0]))
    print(chance)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[1]))
    print(chance)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[2]))
    print(chance)
  
    # 从adjusted=False到adjusted=True的转换
    print(adjusted == (balanced - chance) / (1 - chance) )
    print("+++++++++++++")
    # 采用不平衡测试集（adjusted=True），则平衡准确率为0
    print(balanced_accuracy_score(y_true, np.full_like(y_true, y_true[6]), adjusted=False))
    # 采用不平衡测试集（adjusted=True），则平衡准确率为1/(n_classes)
    chance = balanced_accuracy_score(y_true, np.full_like(y_true, y_true[6]), adjusted=True)
    print(chance)
    print("-------------")
    y_true = [0, 1, 2, 0]
    y_pred = [1, 2, 0, 1]
    # 完全错误的数据集（adjusted=True），则平衡准确率为1/(1 - n_classes)
    print(balanced_accuracy_score(y_true, y_pred, adjusted=True))
    # 完全错误的数据集（adjusted=True），则平衡准确率为0
    print(balanced_accuracy_score(y_true, y_pred, adjusted=False))

test_balanced_accuracy_score()
复制代码
```

运行结果：

```
0.5416666666666666
0.38888888888888884
-------------
[0 0 0 0 0 0 0]
[1 1 1 1 1 1 1]
[2 2 2 2 2 2 2]
[4 4 4 4 4 4 4]
0.25
0.25
0.25
True
+++++++++++++
0.25
0.0
-------------
-0.49999999999999994
0.0
复制代码
```

## 总结

| 函数                       | 说明                                                             |
| -------------------------- | ---------------------------------------------------------------- |
| `accuracy_score`         | 适用于二分类、多分类和多标签分类场景。通常用于平衡数据集的场景。 |
| `top_k_accuracy_score`   | 适用于二分类、多分类场景。                                       |
| `balance_accuracy_score` | 适用于二分类、多分类场景。通常用于不平衡数据集的场景。           |

## 参考文档

* [accuracy-score](https://link.juejin.cn/?target=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fmodel_evaluation.html%23accuracy-score "https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score")
* [top-k-accuracy-score](https://link.juejin.cn/?target=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fmodel_evaluation.html%23top-k-accuracy-score "https://scikit-learn.org/stable/modules/model_evaluation.html#top-k-accuracy-score")
* [balanced-accuracy-score](https://link.juejin.cn/?target=https%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fmodel_evaluation.html%23balanced-accuracy-score "https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score")
* [sklearn.metrics.balanced_accuracy_score样例](https://link.juejin.cn/?target=https%3A%2F%2Fwww.programcreek.com%2Fpython%2Fexample%2F120042%2Fsklearn.metrics.balanced_accuracy_score "https://www.programcreek.com/python/example/120042/sklearn.metrics.balanced_accuracy_score")
