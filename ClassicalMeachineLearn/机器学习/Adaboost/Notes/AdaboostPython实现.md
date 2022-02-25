# AdaBoost算法详解与python实现

目前网上大多数博客只介绍了AdaBoost算法是什么，但是鲜有人介绍为什么Adaboost长这样，本文对此给出了详细的解释。

> 本文首发于我的博客，知乎排版可能有问题，所以建议直接看我的[博客](https://link.zhihu.com/?target=https%3A//tangshusen.me/2018/11/18/adaboost/)

---

## 1. 概述

## 1.1 集成学习

目前存在各种各样的机器学习算法，例如SVM、决策树、感知机等等。但是实际应用中，或者说在打比赛时，成绩较好的队伍几乎都用了集成学习(ensemble learning)的方法。集成学习的思想，简单来讲，就是“三个臭皮匠顶个诸葛亮”。集成学习通过结合多个学习器(例如同种算法但是参数不同，或者不同算法)，一般会获得比任意单个学习器都要好的性能，尤其是在这些学习器都是"弱学习器"的时候提升效果会很明显。

> 弱学习器指的是性能不太好的学习器，比如一个准确率略微超过50%的二分类器。

下面看看西瓜书对此做的一个简单理论分析。
考虑一个二分类问题![[公式]](https://www.zhihu.com/equation?tex=y+%5Cin+%5C%7B-1%2C+%2B1%5C%7D)、真实函数![[公式]](https://www.zhihu.com/equation?tex=f)以及奇数![[公式]](https://www.zhihu.com/equation?tex=M)个犯错概率**相互独立**且均为![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon)的个体学习器(或者称基学习器)![[公式]](https://www.zhihu.com/equation?tex=h_i)。我们用简单的投票进行集成学习，即分类结果取半数以上的基学习器的结果:![[公式]](https://www.zhihu.com/equation?tex=H%28x%29+%3D+sign%28%5Csum_%7Bi%3D1%7D%5EM+h_i%28x%29%29+%5Ctag%7B1.1.1%7D)

由Hoeffding不等式知，集成学习后的犯错(即过半数基学习器犯错)概率满足![[公式]](https://www.zhihu.com/equation?tex=P%28H%28x%29+%5Cneq+f%28x%29%29+%5Cleq+exp%28-+%5Cfrac+1+2+M+%281-2%5Cepsilon%29%5E2%29+%5Ctag%7B1.1.2%7D)

> 式![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%881.1.2%EF%BC%89)的证明是周志华《机器学习》的习题8.1，题解可参考[此处](https://link.zhihu.com/?target=https%3A//blog.csdn.net/icefire_tyh/article/details/52194771)

式![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%881.1.2%EF%BC%89)指出，当**犯错概率**独立的基学习器个数![[公式]](https://www.zhihu.com/equation?tex=M)很大时，集成后的犯错概率接近0，这也很符合直观想法: 大多数人同时犯错的概率是比较低的。

就如上面加粗字体强调的，以上推论全部建立在基学习器犯错相互独立的情况下，但实际中这些学习器不可能相互独立，而如何让基学习器变得“相对独立一些”，也即增加这些基学习器的多样性，正是集成学习需要考虑的主要问题。

按照每个基学习器之间是否存在依赖关系可以将集成学习分为两类：

1. 基学习器之间存在强依赖关系，一系列基学习器需要串行生成，代表算法是 **Boosting** ；
2. 基学习器之间不存在强依赖关系，一系列基学习器可并行生成，代表算法是**Bagging**和 **随机森林** 。

Boosting系列算法里最著名算法主要有AdaBoost和提升树(Boosting tree)系列算法，本文只介绍最具代表性的AdaBoost。提升树、Bagging以及随机森林不在本文介绍范围内，有时间了再另外介绍。

## 1.2 Boosting

Boosting指的是一类集成方法，其主要思想就是将弱的基学习器提升(boost)为强学习器。具体步骤如下:

1. 先用每个样本权重相等的训练集训练一个初始的基学习器；
2. 根据上轮得到的学习器对训练集的预测表现情况调整训练集中的样本权重(例如提高被错分类的样本的权重使之在下轮训练中得到更多的关注), 然后据此训练一个新的基学习器；
3. 重复2直到得到![[公式]](https://www.zhihu.com/equation?tex=M)个基学习器，最终的集成结果是![[公式]](https://www.zhihu.com/equation?tex=M)个基学习器的组合。

由此看出，Boosting算法是一个串行的过程。

Boosting算法簇中最著名的就是AdaBoost，下文将会详细介绍。

## 2. AdaBoost原理

## 2.1 基本思想

对于1.2节所述的Boosting算法步骤，需要回答两个问题:

1. 如何调整每一轮的训练集中的样本权重？
2. 如何将得到的![[公式]](https://www.zhihu.com/equation?tex=M)个组合成最终的学习器？

AdaBoost(Adaptive Boosting, 自适应增强)算法采取的方法是:

1. 提高上一轮被错误分类的样本的权值，降低被正确分类的样本的权值；
2. 线性加权求和。误差率小的基学习器拥有较大的权值，误差率大的基学习器拥有较小的权值。

Adaboost算法结构如下图([图片来源](https://link.zhihu.com/?target=https%3A//www.python-course.eu/Boosting.php))所示。

![](https://pic1.zhimg.com/80/v2-4519fb599b303b6b8ba8b3bd18431384_1440w.jpg)

下面先给出AdaBoost算法具体实现步骤，至于算法解释（为什么要这样做）将在下一大节阐述。

## 2.2 算法步骤

考虑如下形式的二分类（标准AdaBoost算法只适用于二分类任务）训练数据集:![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%28x_1%2Cy_1%29%2C%28x_2%2Cy_2%29%2C...%2C%28x_N%2Cy_N%29%5C%7D)其中![[公式]](https://www.zhihu.com/equation?tex=x_i)是一个含有![[公式]](https://www.zhihu.com/equation?tex=d)个元素的列向量, 即![[公式]](https://www.zhihu.com/equation?tex=x_i%5Cin+%5Cmathcal%7BX%7D+%5Csubseteq+%5Cmathbf%7BR%7D%5Ed);![[公式]](https://www.zhihu.com/equation?tex=y_i)是标量,![[公式]](https://www.zhihu.com/equation?tex=y%5Cin%5C%7B%2B1%2C-1%5C%7D)。

Adaboost算法具体步骤如下:

1. 初始化样本的权重![[公式]](https://www.zhihu.com/equation?tex=D_1%3D%28w_%7B11%7D%2C+w_%7B12%7D%2C...w_%7B1N%7D%29%2C+w_%7B1i%7D%3D%5Cfrac+1+N%2C+i+%3D+1%2C2...N+%5Ctag%7B2.2.1%7D)
2. 对![[公式]](https://www.zhihu.com/equation?tex=m+%3D+1%2C2%2C...M),重复以下操作得到![[公式]](https://www.zhihu.com/equation?tex=M)个基学习器:
   (1) 按照样本权重分布![[公式]](https://www.zhihu.com/equation?tex=D_m)训练数据得到第![[公式]](https://www.zhihu.com/equation?tex=m)个基学习器![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29):![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29%3A+%5Cmathcal%7BX%7D+%5Cto+%5C%7B-1%2C+%2B1%5C%7D)(2) 计算![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)在加权训练数据集上的分类误差率:![[公式]](https://www.zhihu.com/equation?tex=e_m+%3D+%5Csum_%7Bi%3D1%7D%5ENP%28G_m%28x_i%29+%5Cneq+y_i%29%3D%5Csum_%7Bi%3D1%7D%5EN+w_%7Bmi%7D+I%28G_m%28x_i%29+%5Cneq+y_i%29+%5Ctag%7B2.2.2%7D) 上式中![[公式]](https://www.zhihu.com/equation?tex=I%28%5Ccdot%29)是指示函数，考虑更加周全的AdaBoost算法在这一步还应该判断是否满足基本条件(例如生成的基学习器是否比随机猜测好), 如果不满足，则当前基学习器被抛弃，学习过程提前终止。
   (3) 计算![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)的系数(即最终集成使用的的基学习器的权重):![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m+%3D+%5Cfrac+1+2+log+%5Cfrac+%7B1-e_m%7D+%7Be_m%7D+%5Ctag%7B2.2.3%7D) (4) 更新训练样本的权重![[公式]](https://www.zhihu.com/equation?tex=D_%7Bm%2B1%7D%3D%28w_%7Bm%2B1%2C1%7D%2C+w_%7Bm%2B1%2C2%7D%2C...w_%7Bm%2B1%2CN%7D%29+%5Ctag%7B2.2.4%7D)![[公式]](https://www.zhihu.com/equation?tex=w_%7Bm%2B1%2C+i%7D+%3D+%5Cfrac%7Bw_%7Bmi%7D%7D+%7BZ_m%7D+exp%28-%5Calpha_my_iG_m%28x_i%29%29+%2Ci%3D1%2C2%2C...N+%5Ctag%7B2.2.5%7D) 其中![[公式]](https://www.zhihu.com/equation?tex=Z_m)是规范化因子，目的是为了使![[公式]](https://www.zhihu.com/equation?tex=D_%7Bm%2B1%7D)的所有元素和为1。即![[公式]](https://www.zhihu.com/equation?tex=Z_m%3D%5Csum_%7Bi%3D1%7D%5EN+w_%7Bmi%7D+exp%28-%5Calpha_my_iG_m%28x_i%29%29+%5Ctag%7B2.2.6%7D)
3. 构建最终的分类器线性组合![[公式]](https://www.zhihu.com/equation?tex=f%28x%29+%3D+%5Csum_%7Bi%3D1%7D%5EM+%5Calpha_m+G_m%28x%29+%5Ctag%7B2.2.7%7D) 得到最终的分类器为![[公式]](https://www.zhihu.com/equation?tex=G%28x%29+%3D+sign%28f%28x%29%29%3Dsign%28%5Csum_%7Bi%3D1%7D%5EM+%5Calpha_m+G_m%28x%29%29+%5Ctag%7B2.2.8%7D)

由式![[公式]](https://www.zhihu.com/equation?tex=%282.2.3%29)知，当基学习器![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)的误差率![[公式]](https://www.zhihu.com/equation?tex=e_m+%5Cle+0.5)时，![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m+%5Cge+0)，并且![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)随着![[公式]](https://www.zhihu.com/equation?tex=e_m)的减小而增大，即分类误差率越小的基学习器在最终集成时占比也越大。即AdaBoost能够适应各个弱分类器的训练误差率，这也是它的名称中"适应性(Adaptive)"的由来。

由式![[公式]](https://www.zhihu.com/equation?tex=%282.2.5%29)知， 被基学习器![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)误分类的样本权值得以扩大，而被正确分类的样本的权值被得以缩小。

需要注意的是式![[公式]](https://www.zhihu.com/equation?tex=%282.2.7%29)中所有的![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)的和并不为1(因为没有做一个softmax操作)，![[公式]](https://www.zhihu.com/equation?tex=f%28x%29)的符号决定了所预测的类，其绝对值代表了分类的确信度。

## 3. AdaBoost算法解释

有没有想过为什么AdaBoost算法长上面这个样子，例如为什么![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)要用式![[公式]](https://www.zhihu.com/equation?tex=%282.2.3%29)那样计算？本节将探讨这个问题。

## 3.1 前向分步算法

在解释AdaBoost算法之前，先来看看前向分步算法。

就以AdaBoost算法的最终模型表达式为例:![[公式]](https://www.zhihu.com/equation?tex=f%28x%29+%3D+%5Csum_%7Bi%3D1%7D%5EM+%5Calpha_m+G_m%28x%29+%5Ctag%7B3.1.1%7D) 可以看到这是一个“加性模型(additive model)”。我们希望这个模型在训练集上的经验误差最小，即

![[公式]](https://www.zhihu.com/equation?tex=min+%5Csum_%7Bi%3D1%7D%5EN+L%28y_i%2C+f%28x%29%29+%5Ciff+min+%5Csum_%7Bi%3D1%7D%5EN+L%28y_i%2C+%5Csum_%7Bi%3D1%7D%5EM+%5Calpha_m+G_m%28x%29%29+%5Ctag%7B3.1.2%7D) 通常这是一个复杂的优化问题。前向分步算法求解这一优化问题的思想就是: 因为最终模型是一个加性模型，如果能从前往后，每一步只学习一个基学习器![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)及其权重![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m), 不断迭代得到最终的模型，那么就可以简化问题复杂度。具体的，当我们经过![[公式]](https://www.zhihu.com/equation?tex=m-1)轮迭代得到了最优模型![[公式]](https://www.zhihu.com/equation?tex=f_%7Bm-1%7D%28x%29)时，因为

![[公式]](https://www.zhihu.com/equation?tex=f_m%28x%29%3D+f_%7Bm-1%7D%28x%29+%2B+%5Calpha_mG_m%28x%29+%5Ctag%7B3.1.3%7D) 所以此轮优化目标就为![[公式]](https://www.zhihu.com/equation?tex=min+%5Csum_%7Bi%3D1%7D%5EN+L%28y_i%2C+f_%7Bm-1%7D%28x%29+%2B+%5Calpha_mG_m%28x%29%29+%5Ctag%7B3.1.4%7D) 求解上式即可得到第![[公式]](https://www.zhihu.com/equation?tex=m)个基分类器![[公式]](https://www.zhihu.com/equation?tex=G_m%28x%29)及其权重![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)。
这样，前向分步算法就通过不断迭代求得了从![[公式]](https://www.zhihu.com/equation?tex=m%3D1)到![[公式]](https://www.zhihu.com/equation?tex=m%3DM)的所有基分类器及其权重，问题得到了解决。

## 3.2 AdaBoost算法证明

上一小结介绍的前向分步算法逐一学习基学习器，这一过程也即AdaBoost算法逐一学习基学习器的过程。但是为什么2.2节中的公式为什么长那样还是没有解释。本节就证明前向分步算法的损失函数是指数损失函数(exponential loss function)时，AdaBoost学习的具体步骤就如2.2节所示。

> 指数损失函数即![[公式]](https://www.zhihu.com/equation?tex=L%28y%2C+f%28x%29%29+%3D+exp%28-yf%28x%29%29)周志华《机器学习》p174有证明，指数损失函数是分类任务原本0/1损失函数的一致(consistent)替代损失函数，由于指数损失函数有更好的数学性质，例如处处可微，所以我们用它替代0/1损失作为优化目标。

将指数损失函数代入式![[公式]](https://www.zhihu.com/equation?tex=%283.1.4%29)，优化目标就为![[公式]](https://www.zhihu.com/equation?tex=%5Cunderset%7B%5Calpha_m%2CG_m%7D%7Bargmin%7D+%5Csum_%7Bi%3D1%7D%5EN+exp%5B-y_i%28f_%7Bm-1%7D%28x%29+%2B+%5Calpha_mG_m%28x%29%29%5D+%5Ctag%7B3.2.1%7D) 因为![[公式]](https://www.zhihu.com/equation?tex=y_if_%7Bm-1%7D%28x%29)与优化变量![[公式]](https://www.zhihu.com/equation?tex=%5Calpha)和![[公式]](https://www.zhihu.com/equation?tex=G)无关，如果令![[公式]](https://www.zhihu.com/equation?tex=w_%7Bm%2Ci%7D+%3D+exp%5B-y_i+f_%7Bm-1%7D%28x%29%5D+%5Ctag%7B3.2.2%7D)

> 这个![[公式]](https://www.zhihu.com/equation?tex=w_%7Bm%2Ci%7D)其实就是2.2节中归一化之前的权重![[公式]](https://www.zhihu.com/equation?tex=w_%7Bm%2Ci%7D)

那么式![[公式]](https://www.zhihu.com/equation?tex=%283.2.1%29)等价于![[公式]](https://www.zhihu.com/equation?tex=%5Cunderset%7B%5Calpha_m%2CG_m%7D%7Bargmin%7D+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7Dexp%28-y_i%5Calpha_mG_m%28x%29%29+%5Ctag%7B3.2.3%7D)

我们分两步来求解式![[公式]](https://www.zhihu.com/equation?tex=%283.2.3%29)所示的优化问题的最优解![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Calpha%7D_m)和![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BG%7D_m%28x%29):

1. 对任意的![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m+%3E+0), 求![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BG%7D_m%28x%29)：![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BG%7D_m+%28x%29+%3D+%5Cunderset%7BG_m%7D%7Bargmin%7D+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+I%28y_i+%5Cneq+G_m%28x_i%29%29+%5Ctag%7B3.2.4%7D) 上式将指数函数换成指示函数是因为前面说的指数损失函数和0/1损失函数是一致等价的。
   式子![[公式]](https://www.zhihu.com/equation?tex=%283.2.4%29)所示的优化问题其实就是AdaBoost算法的基学习器的学习过程，即2.2节的步骤2(1)，得到的![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BG%7D_m%28x%29)是使第![[公式]](https://www.zhihu.com/equation?tex=m)轮加权训练数据分类误差最小的基分类器。
2. 求解![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Calpha%7D_m)：
   将式子![[公式]](https://www.zhihu.com/equation?tex=%283.2.3%29)中的目标函数展开![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7Dexp%28-y_i%5Calpha_mG_m%28x%29%29+%26%3D+%5Csum_%7By_i%3DG_m%28x_i%29%7D+w_%7Bm%2Ci%7De%5E%7B-+%5Calpha%7D+%2B+%5Csum_%7By_i+%5Cneq+G_m%28x_i%29%7Dw_%7Bm%2Ci%7De%5E%7B%5Calpha%7D+%5C%5C%5C%5C+%26+%3D+%28e%5E%7B%5Calpha%7D+-+e%5E%7B-%5Calpha%7D%29+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+I%28y_i+%5Cneq+G_m%28x_i%29%29+%2B+e%5E%7B-%5Calpha%7D+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+%5Cend%7Baligned%7D+%5Ctag%7B3.2.5%7D)注：为了简洁，上式子中的![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BG%7D_m%28x%29)被略去了![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ccdot%7D)，![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)被略去了下标![[公式]](https://www.zhihu.com/equation?tex=m)，下同
   将上式对![[公式]](https://www.zhihu.com/equation?tex=%5Calpha)求导并令导数为0，即![[公式]](https://www.zhihu.com/equation?tex=%28e%5E%7B%5Calpha%7D+%2B+e%5E%7B-%5Calpha%7D%29+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+I%28y_i+%5Cneq+G_m%28x_i%29%29+-+e%5E%7B-%5Calpha%7D+%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+%3D+0+%5Ctag%7B3.2.6%7D) 解得![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Calpha%7D_m+%3D+%5Cfrac+1+2+log+%5Cfrac+%7B1-e_m%7D+%7Be_m%7D+%5Ctag%7B3.2.7%7D) 其中,![[公式]](https://www.zhihu.com/equation?tex=e_m)是分类误差率：![[公式]](https://www.zhihu.com/equation?tex=e_m+%3D+%5Cfrac+%7B%5Csum_%7Bi%3D1%7D%5EN+w_%7Bm%2Ci%7D+I%28y_i+%5Cneq+G_m%28x_i%29%7D+%7B%5Csum_%7Bi%3D1%7D%5EN+w_%7Bmi%7D%7D+%5Ctag%7B3.2.8%7D) 如果式子![[公式]](https://www.zhihu.com/equation?tex=%283.2.8%29)中的![[公式]](https://www.zhihu.com/equation?tex=w_%7Bmi%7D)归一化成和为1的话那么式![[公式]](https://www.zhihu.com/equation?tex=%283.2.8%29)也就和2.2节式![[公式]](https://www.zhihu.com/equation?tex=%282.2.2%29)一模一样了，进一步地也有上面的![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Calpha%7D_m)也就是2.2节的![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_m)。
   最后来看看每一轮样本权值的更新，由![[公式]](https://www.zhihu.com/equation?tex=%283.1.3%29)和![[公式]](https://www.zhihu.com/equation?tex=%283.2.2%29)可得![[公式]](https://www.zhihu.com/equation?tex=w_%7Bm%2B1%2Ci%7D+%3D+w_%7Bm%2Ci%7D+exp%5B-y_i+%5Calpha_m+G_%7Bm%7D%28x%29%5D+%5Ctag%7B3.2.9%7D) 如果将上式进行归一化成和为1的话就和与2.2节中![[公式]](https://www.zhihu.com/equation?tex=%282.2.5%29)完全相同了。

由此可见，2.2节所述的AdaBoost算法步骤是可以经过严密推导得来的。总结一下，本节推导有如下关键点:

* AdaBoost算法是一个加性模型，将其简化成 **前向分步算法求解** ；
* 将0/1损失函数用数学性质更好的**指数损失函数**替代。

## 4. python实现

## 4.1 基学习器

首先需要定义一个基学习器，它应该是一个弱分类器。
弱分类器使用库 `sklearn`中的决策树分类器 `DecisionTreeClassifier`, 可设置该决策树的最大深度为1。

```python
# Fit a simple decision tree(weak classifier) first
clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
```

## 4.2 AdaBoost实现

然后就是完整AdaBoost算法的实现了，如下所示。

```python
def my_adaboost_clf(Y_train, X_train, Y_test, X_test, M=20, weak_clf=DecisionTreeClassifier(max_depth = 1)):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
         % (i + 1, 1 - sum(miss) / n_train))

        # Error
        err_m = np.dot(w, miss)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    print("My AdaBoost clf test accuracy: %.4f" % (sum(pred_test == Y_test) / n_test
```
