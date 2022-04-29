# 什么是ROC曲线？为什么要使用ROC?以及 AUC的计算

## 一、ROC简介

> ROC的全名叫做Receiver Operating Characteristic，中文名字叫“ **受试者工作特征曲线** ”，其主要分析工具是一个画在二维平面上的曲线——ROC 曲线。平面的横坐标是false positive rate(FPR)，纵坐标是true positive rate(TPR)。对某个分类器而言，我们可以根据其在**测试样本上**的表现得到一个 **TPR和FPR点对** 。这样，此分类器就可以映射成ROC平面上的一个点。调整这个分类器分类时候使用的阈值，我们就可以得到一个经过(0, 0)，(1, 1)的曲线，这就是此分类器的ROC曲线。
> 一般情况下，这个曲线都应该处于(0, 0)和(1, 1) **连线的上方** 。因为(0, 0)和(1, 1)连线形成的ROC曲线实际上代表的是一个随机分类器。如果很不幸，你得到一个位于此直线下方的分类器的话，一个直观的补救办法就是把所有的预测结果反向，即：分类器输出结果为正类，则最终分类的结果为负类，反之，则为正类。虽然，用ROC 曲线来表示分类器的性能很直观好用。`可是，人们总是希望能有一个数值来标志分类器的好坏。于是` **`Area Under roc Curve(AUC)`** `就出现了。顾名思义，AUC的值就是处于` **`ROC 曲线下方的那部分面积的大小`** `。`通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的性能。AUC（Area Under roc Curve）是一种用来度量分类模型好坏的一个标准。

![](https://ask.qcloudimg.com/http-save/yehe-7969553/hirzm1wjyv.png?imageView2/2/w/1620)

## 二、基本概念

解读ROC图的一些概念定义:：

### 1. 四种分类

 真正（True Positive , TP）被模型预测为正的正样本；
 假负（False Negative , FN）被模型预测为负的正样本；
 假正（False Positive , FP）被模型预测为正的负样本；
 真负（True Negative , TN）被模型预测为负的负样本。

### 2. 横纵坐标解释

该曲线的横坐标为假阳性率（False Positive Rate, FPR），N是真实负样本的个数，
 FP是N个负样本中被分类器预测为正样本的个数。

纵坐标为真阳性率（True Positive Rate, TPR），

![](https://ask.qcloudimg.com/http-save/yehe-7969553/z0od0221uz.png?imageView2/2/w/1620)

 P是真实正样本的个数，
 TP是P个正样本中被分类器预测为正样本的个数。

### 3.混淆矩阵

对于二分类问题，可将样本根据其真实类别与学习器预测类别的组合划分为TP(true positive)、FP(false positive)、TN(true negative)、FN(false negative)四种情况，TP+FP+TN+FN=样本总数。

![](https://ask.qcloudimg.com/http-save/yehe-7969553/3p7szxgq9u.png?imageView2/2/w/1620)

### 三、为什么要选择ROC？

既然已经这么多评价标准，为什么还要使用ROC和AUC呢？因为ROC曲线有个很好的特性：当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变。在实际的数据集中经常会出现类不平衡（class imbalance）现象，即负样本比正样本多很多（或者相反），而且测试数据中的正负样本的分布也可能随着时间变化。下图是ROC曲线和Precision-Recall曲线的对比：

![](https://ask.qcloudimg.com/http-save/yehe-7969553/2lmhoffvso.png?imageView2/2/w/1620)

其中第一行ab均为原数据的图，左边为ROC曲线，右边为P-R曲线。第二行cd为负样本增大10倍后俩个曲线的图。可以看出，ROC曲线基本没有变化，但P-R曲线确剧烈震荡。因此，在面对正负样本数量不均衡的场景下，ROC曲线（AUC的值）会是一个更加稳定能反映模型好坏的指标。

### 四、AUC作为评价标准

**1. AUC (Area Under Curve)**

被定义为ROC曲线下的面积，取值范围一般在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

**2.AUC 的计算方法**

非参数法：（两种方法实际证明是一致的）

(1) *梯形法则* ：早期由于测试样本有限，我们得到的AUC曲线呈阶梯状。曲线上的每个点向X轴做垂线，得到若干梯形，这些梯形面积之和也就是AUC 。

(2) *Mann-Whitney统计量* ： 统计正负样本对中，有多少个组中的正样本的概率大于负样本的概率。这种估计随着样本规模的扩大而逐渐逼近真实值。

参数法：

(3)主要适用于*二项分布*的数据，即正反样本分布符合正态分布，可以通过均值和方差来计算。

**3.从AUC判断分类器（预测模型）优劣的标准**

· AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。

· 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。

· AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。

· AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

三种AUC值示例：

![](https://ask.qcloudimg.com/http-save/yehe-7969553/3te9r5la91.png?imageView2/2/w/1620)

**总结：AUC值越大的分类器，正确率越高**

**4. 不同模型AUC的比较**

总的来说，AUC值越大，模型的分类效果越好，疾病检测越准确；不过两个模型AUC值相等并不代表模型效果相同，例子如下：

下图中有三条ROC曲线，A模型比B和C都要好

![](https://ask.qcloudimg.com/http-save/yehe-7969553/gslgf0jkhp.png?imageView2/2/w/1620)

下面两幅图中两条ROC曲线相交于一点，AUC值几乎一样：当需要高Sensitivity时，模型A比B好；当需要高Speciticity时，模型B比A好

![](https://ask.qcloudimg.com/http-save/yehe-7969553/6u83lfx6lk.png?imageView2/2/w/1620)

![](https://ask.qcloudimg.com/http-save/yehe-7969553/lhj5usm72h.png?imageView2/2/w/1620)
