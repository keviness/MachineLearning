# 机器学习基础（1）- ROC曲线理解

本文用于理解ROC曲线的定义，绘制过程及其应用实现，主要用于自我温习回顾基础

基本目录如下：

1. 什么是ROC曲线？
   1.1 ROC曲线的历史
   1.2 ROC曲线的定义
   1.3 ROC曲线的应用场景
2. 如何绘制ROC曲线？
   2.1 ROC曲线的绘制原理
   2.2 ROC曲线绘制的Python实现

------------------第一菇 - 什么是ROC曲线------------------

##### 1.1 ROC曲线的历史

自从读了吴军老师的《数学之美》，我就想明白了一件事情，如果想要讲明白一件事情，一定要把他的历史渊源都讲明白，这样我们才能对其理解透彻，而不是单纯学到会用就好～试想，有多少人在读这篇文章之前，会想到ROC曲线在军事上的运用呢？接下来，我就当一回搬运工，把ROC曲线的诞生渊源都捋一捋～

经过一番网上调查，ROC曲线起源于第二次世界大战时期雷达兵对雷达的信号判断【1】。当时每一个雷达兵的任务就是去解析雷达的信号，但是当时的雷达技术还没有那么先进，存在很多噪声（比如一只大鸟飞过），所以每当有信号出现在雷达屏幕上，雷达兵就需要对其进行破译。有的雷达兵比较谨慎，凡是有信号过来，他都会倾向于解析成是敌军轰炸机，有的雷达兵又比较神经大条，会倾向于解析成是飞鸟。这个时候，雷达兵的上司就很头大了，他急需一套评估指标来帮助他汇总每一个雷达兵的预测信息，以及来评估这台雷达的可靠性（如果不论哪一类雷达兵都能准确预测，那这台雷达就很NB～读者可思考其缘由）。于是，最早的ROC曲线分析方法就诞生了，用来作为评估雷达可靠性的指标～在那之后，ROC曲线就被广泛运用于医学以及机器学习领域～

##### 1.2 ROC曲线的定义

ROC的全称是Receiver Operating Characteristic Curve，中文名字叫“受试者工作特征曲线”，顾名思义，其主要的分析方法就是画这条特征曲线。这里在网上找了一个比较好的图样示例如下，

![](https://upload-images.jianshu.io/upload_images/11525720-7eedb3ee87fa4111.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/519/format/webp)

ROC曲线示例

该曲线的横坐标为假阳性率（False Positive Rate, FPR），N是真实负样本的个数，
FP是N个负样本中被分类器预测为正样本的个数。

纵坐标为真阳性率（True Positive Rate, TPR），
![TPR=\frac{TP}{P}](https://math.jianshu.com/math?formula=TPR%3D%5Cfrac%7BTP%7D%7BP%7D)

P是真实正样本的个数，
TP是P个正样本中被分类器预测为正样本的个数。

举一个简单的例子方便大家的理解，还是刚才雷达的例子。假设现在有10个雷达信号警报，其中8个是真的轰炸机（P）来了，2个是大鸟（N）飞过，经过某分析员解析雷达的信号，判断出9个信号是轰炸机，剩下1个是大鸟，其中被判定为轰炸机的信号中，有1个其实是大鸟的信号（FP=1），而剩下8个确实是轰炸机信号（TP=8）。因此可以计算出FPR为![0.5](https://math.jianshu.com/math?formula=0.5)，TPR为![1](https://math.jianshu.com/math?formula=1)，而![(0.5,1)](https://math.jianshu.com/math?formula=(0.5%2C1))就对应ROC曲线上一点。

说到这里，想必大家已经明白这俩个指标的计算方法，再往深挖一点，可以思考一下这俩个指标背后的原理。还是雷达的例子，敏锐的雷达系统我们肯定希望它能把所有的敌方轰炸机来袭都感知到并预测出来，即TPR越高越好，但我们又不希望它把大鸟的飞过也当成轰炸机来预警，即FRP越低越好。因此，大家可以发现，这俩个坐标值其实是有相互制约的一个概念在里面。

当绘制完成曲线后，就会对模型有一个定性的分析，如果要对模型进行量化的分析，此时需要引入一个新的概念，就是AUC（Area under roc Curve）面积，这个概念其实很简单，就是指ROC曲线下的面积大小，而计算AUC值只需要沿着ROC横轴做积分就可以了。真实场景中ROC曲线一般都会在![y=x](https://math.jianshu.com/math?formula=y%3Dx)这条直线的上方，所以AUC的取值一般在0.5~1之间。AUC的值越大，说明该模型的性能越好。

##### 1.3 ROC曲线的应用场景

ROC曲线的应用场景有很多，根据上述的定义，其最直观的应用就是能反映模型在选取不同阈值的时候其敏感性（sensitivity, FPR）和其精确性（specificity, TPR）的趋势走向【2】。不过，相比于其他的P-R曲线（精确度和召回率），ROC曲线有一个巨大的优势就是，当正负样本的分布发生变化时，其形状能够基本保持不变，而P-R曲线的形状一般会发生剧烈的变化，因此该评估指标能降低不同测试集带来的干扰，更加客观的衡量模型本身的性能。要解释清楚这个问题的话，大家还是先回顾一下混淆矩阵。

![](https://upload-images.jianshu.io/upload_images/11525720-7131daae2ff90acb.png?imageMogr2/auto-orient/strip|imageView2/2/w/621/format/webp)

混淆矩阵

其中，精确率P的计算公式为，

召回率R的计算公式为，

此时，若将负样本的数量增加，扩大个10倍，可以预见FP,TN都会增加，必然会影响到P,R。但ROC曲线的俩个值，FPR只考虑第二行，N若增大10倍，则FP,TN也会成比例增加，并不影响其值，TPR更是只考虑第一行，不会受到影响。这里在网上盗个图【3】，方便大家理解哈～

![](https://upload-images.jianshu.io/upload_images/11525720-311f6fb4e809447d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1167/format/webp)

ROC曲线和P-R曲线对比图

其中第一行ab均为原数据的图，左边为ROC曲线，右边为P-R曲线。第二行cd为负样本增大10倍后俩个曲线的图。可以看出，ROC曲线基本没有变化，但P-R曲线确剧烈震荡。因此，在面对正负样本数量不均衡的场景下，ROC曲线（AUC的值）会是一个更加稳定能反映模型好坏的指标。

------------------第二菇 - 如何绘制ROC曲线------------------

##### 2.1 ROC曲线的绘制原理

如果大家对二值分类模型熟悉的话，都会知道其输出一般都是预测样本为正例的概率，而事实上，ROC曲线正是通过不断移动分类器的“阈值”来生成曲线上的一组关键点的。可能这样讲有点抽象，还是举刚才雷达兵的例子。每一个雷达兵用的都是同一台雷达返回的结果，但是每一个雷达兵内心对其属于敌军轰炸机的判断是不一样的，可能1号兵解析后认为结果大于0.9，就是轰炸机，2号兵解析后认为结果大于0.85，就是轰炸机，依次类推，每一个雷达兵内心都有自己的一个判断标准（也即对应分类器的不同“阈值”），这样针对每一个雷达兵，都能计算出一个ROC曲线上的关键点（一组FPR,TPR值），把大家的点连起来，也就是最早的ROC曲线了。

为方便大家进一步理解，本菇也在网上找到了一个示例跟大家一起分享【4】。下图是一个二分模型真实的输出结果，一共有20个样本，输出的概率就是模型判定其为正例的概率，第二列是样本的真实标签。

![](https://upload-images.jianshu.io/upload_images/11525720-cb0c836e33757b87.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

二值分类模型输出结果示例

现在我们指定一个阈值为0.9，那么只有第一个样本（0.9）会被归类为正例，而其他所有样本都会被归为负例，因此，对于0.9这个阈值，我们可以计算出FPR为0，TPR为0.1（因为总共10个正样本，预测正确的个数为1），那么我们就知道曲线上必有一个点为(0, 0.1)。依次选择不同的阈值（或称为“截断点”），画出全部的关键点以后，再连接关键点即可最终得到ROC曲线如下图所示。

![](https://upload-images.jianshu.io/upload_images/11525720-dd2545eaaaa7c2ba.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

ROC曲线示例图

其实还有一种更直观的绘制ROC曲线的方法，这边简单提一下。就是把横轴的刻度间隔设为![\frac{1}{N}](https://math.jianshu.com/math?formula=%5Cfrac%7B1%7D%7BN%7D)，纵轴的刻度间隔设为![\frac{1}{P}](https://math.jianshu.com/math?formula=%5Cfrac%7B1%7D%7BP%7D)，N,P分别为负样本与正样本数量。然后再根据模型的输出结果降序排列，依次遍历样本，从0开始绘制ROC曲线，每遇到一个正样本就沿纵轴方向绘制一个刻度间隔的曲线，每遇到一个负样本就沿横轴方向绘制一个刻度间隔的曲线，遍历完所有样本点以后，曲线也就绘制完成了。究其根本，其最大的好处便是不需要再去指定阈值寻求关键点了，每一个样本的输出概率都算是一个阈值了。当然，无论是工业界还是学术界的实现，都不可能手动去绘制，下面就来讲一下如何用Python高效绘制ROC曲线。

##### 2.2 ROC曲线绘制的Python实现

熟悉sklearn的读者肯定都知道，几乎所有评估模型的指标都来自sklearn库下面的metrics，包括计算召回率，精确率等。ROC曲线的绘制也不例外，都得先计算出评估的指标，也就是从metrics里面去调用roc_curve, auc，然后再去绘制。

roc_curve和auc的官方说明教程示例如下【5】。

因此调用完roc_curve以后，我们就齐全了绘制ROC曲线的数据。接下来的事情就很简单了，调用plt即可，还是用官方的代码示例一步到底。

最终生成的ROC曲线结果如下图。

![](https://upload-images.jianshu.io/upload_images/11525720-b9141c284217c73a.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

ROC曲线结果

至此，整一套ROC曲线的绘制就说明白了。

简单总结一下本文，先是讲述了ROC曲线的历史渊源，引导读者理解ROC曲线的各个基本概念，又与P-R曲线做了详细的对比，让读者理解其应用场景，最后接地气的轻微解读了一番其绘制过程，并附上了代码实现。希望大家读完本文后对ROC曲线会有一个全新的认识。有说的不对的地方也请大家指出，多多交流，大家一起进步～😁

参考文献：
【1】[https://www.ncbi.nlm.nih.gov/books/NBK22319/](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.ncbi.nlm.nih.gov%2Fbooks%2FNBK22319%2F)
【2】[https://stats.stackexchange.com/questions/28745/advantages-of-roc-curves](https://links.jianshu.com/go?to=https%3A%2F%2Fstats.stackexchange.com%2Fquestions%2F28745%2Fadvantages-of-roc-curves)
【3】[https://blog.csdn.net/songyunli1111/article/details/82285266](https://links.jianshu.com/go?to=https%3A%2F%2Fblog.csdn.net%2Fsongyunli1111%2Farticle%2Fdetails%2F82285266)
【4】[https://www.zhihu.com/question/22844912/answer/246037337](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.zhihu.com%2Fquestion%2F22844912%2Fanswer%2F246037337)
【5】[http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve](https://links.jianshu.com/go?to=http%3A%2F%2Fscikit-learn.org%2Fstable%2Fmodules%2Fgenerated%2Fsklearn.metrics.roc_curve.html%23sklearn.metrics.roc_curve)

224人点赞

[机器学习基础](https://www.jianshu.com/nb/24136361)
