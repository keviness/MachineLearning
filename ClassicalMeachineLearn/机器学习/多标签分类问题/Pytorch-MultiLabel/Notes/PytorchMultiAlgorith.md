我们以NLP中的Ner任务来说，假设模型是Bert，那模型的输出就是每个词的representation，比如是 ![[公式]](https://www.zhihu.com/equation?tex=D%3D%5B128%EF%BC%8C1024%5D) 这样的输出，那么这个时候如果我们需要做标注分类任务的时候，就需要先对这个”表示"进行处理。首先就是要做一个映射，映射到 ![[公式]](https://www.zhihu.com/equation?tex=D+%3D+%5B128%EF%BC%8C10%5D) 的空间，表明是个10分类，然后再进行归一化处理，比如用Softmax（多分类）、Sigmoid（单分类）等操作，最后再通过CrossEntropyLoss这样的函数做损失计算，下面是常用的一些函数的说明。先看下面这张表：

| 分类问题名称 | 输出层使用激活函数 | 对应的损失函数                                                                             |
| ------------ | ------------------ | ------------------------------------------------------------------------------------------ |
| 二分类       | sigmoid函数        | 二分类交叉熵损失函数BCELoss()--不带sigmoid``、BCEWithLogitsLoss()--带sigmoid        |
| 多分类       | softmax函数        | 多类别交叉熵损失函数nn.NLLLoss()--不带LogSoftmax、``nn.CrossEntropy()--带LogSoftmax |
| 多标签分类   | sigmoid函数        | 二分类交叉熵损失函数``BCELoss()、BCEWithLogitsLoss()MultiLabelSoftMarginLoss()      |

## 一、nn.Softmax

首先介绍一下Softmax函数， **Softmax函数输出的是概率分布，** 公式如下： ![[公式]](https://www.zhihu.com/equation?tex=o%28z_i%29%3Dsoftmax%28z_i%29%3D%5Cfrac%7Be%5Ei%7D%7B%5Csum_je%5Ej%7D%5C%5C)

其输出一定是在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%EF%BC%8C1%5D) 之间的，假如输入 ![[公式]](https://www.zhihu.com/equation?tex=x%3D%5B1%EF%BC%8C2%EF%BC%8C13%5D) ，那么Softmax之后的输出就是：

![[公式]](https://www.zhihu.com/equation?tex=x+%3D+%5B%5Cfrac%7Be%5E%7B1%7D%7D%7Be%5E1%2Be%5E2%2Be%5E%7B13%7D%7D%EF%BC%8C%5Cfrac%7Be%5E%7B2%7D%7D%7Be%5E1%2Be%5E2%2Be%5E%7B13%7D%7D%EF%BC%8C%5Cfrac%7Be%5E%7B13%7D%7D%7Be%5E1%2Be%5E2%2Be%5E%7B13%7D%7D%5D+%5Capprox+%5B0%2C0%2C1%5D%5C%5C)

```python
## 代码如下
y = torch.rand(size=[2,3])


##  输出
>> tensor([[0.8101, 0.6255, 0.3686],
           [0.4457, 0.6892, 0.8240]])

net = nn.Softmax(dim=1) ## dim根据具体的情况可选择
output = net(y)

## 输出
>> tensor([[0.4041, 0.3360, 0.2599],
           [0.2677, 0.3415, 0.3908]])
```

nn.Softmax可以用于激活函数或者多分类，其中分类模型如下：

```python
## 下面这段代码主要把softmax函数换成pytorch封装好的nn.Softmax函数即可，
## 不过通常情况下，我们对于多分类问题会直接选择使用nn.CrossEntropy()函数
def softmax(x):
    s = torch.exp(x)
    return s / torch.sum(s, dim=1, keepdim=True)

def crossEntropy(y_true, logits):
    c = -torch.log(logits.gather(1, y_true.reshape(-1, 1)))
    return torch.sum(c)

logits = torch.tensor([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]])
y = torch.LongTensor([2, 1])
c = crossEntropy(y, softmax(logits)) / len(y)
```

## 二、nn.LogSoftmax

相对于nn.Softmax，nn.LogSoftmax多了一个步骤就是对输出进行了log操作，公式如下： ![[公式]](https://www.zhihu.com/equation?tex=o%28z_i%29%3Dlog%28softmax%28z_i%29%29%3Dlog%28%5Cfrac%7Be%5Ei%7D%7B%5Csum_je%5Ej%7D%29%5C%5C)

nn.LogSoftmax输出的是小于0的数。

```python
## 代码如下
y = torch.rand(size=[2,3])

##  输出
>> tensor([[0.8101, 0.6255, 0.3686],
           [0.4457, 0.6892, 0.8240]])

net = nn.LogSoftmax(dim=1) ## dim根据具体的情况可选择
output = net(y)

## 输出
>> tensor([[-0.9060, -1.0907, -1.3475],
           [-1.3179, -1.0744, -0.9396]])
```

nn.LogSoftmax的输出可以用于交叉熵损失函数的计算，对于额二分类和多分类的交叉熵计算公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%26+%28%E4%BA%8C%E5%88%86%E7%B1%BB%29L+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+L_i+%3D+-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D%5By_i%5Ccdot+log%28p_i%29+%2B+%281-y_i%29%5Ccdot+log%281-p_i%29%5D+%5C%5C+%26%28%E5%A4%9A%E5%88%86%E7%B1%BB%29L+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+L_i+%3D+-%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+%5Csum_%7Bc%3D1%7D%5EMy_%7Bic%7D%5Clog%28p_%7Bic%7D%29++%5Cend%7Balign%2A%7D%5C%5C)

其中：

-![[公式]](https://www.zhihu.com/equation?tex=y_i)—— 表示样本i的label，正类为1，负类为0

-![[公式]](https://www.zhihu.com/equation?tex=p_i)—— 表示样本i预测为正的概率

-![[公式]](https://www.zhihu.com/equation?tex=M)——类别的数量；

-![[公式]](https://www.zhihu.com/equation?tex=y_%7Bic%7D)——指示变量（0或1）,如果该类别和样本i的类别相同就是1，否则是0；

-![[公式]](https://www.zhihu.com/equation?tex=p_%7Bic%7D)——对于观测样本i属于类别![[公式]](https://www.zhihu.com/equation?tex=c)的预测概率。

那nn.LogSoftmax就是在计算交叉熵log部分，nn.LogSoftmax()要和nn.NLLLoss()结合使用，NLLLoss()的计算公式如下：

![[公式]](https://www.zhihu.com/equation?tex=a%3D%5Ba_0%2Ca_1%2C...%2Ca_%7BC-1%7D%5D) 表示**一个样本**对每个类别的 **对数似然** (log-probabilities)， ![[公式]](https://www.zhihu.com/equation?tex=c) 表示该 **样本的标签（比如1，2，-100等等）** ，**单个样本**的损失函数公式描述如下:

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+%26loss%28a%2Cc%29+%3D+-w+%5Ccdot+a%5Bc%5D+%3D+-w+log%28p_c%29%5C%5C+%26%E5%85%B6%E4%B8%AD%EF%BC%8C+w+%3D+weight%5Bc%5D+%5Ccdot+1%5C%7Bc+%5Cne+ignore%5C_index+%5C%7D%5C%5C+%5Cend%7Balign%2A%7D%5C%5C+)

```python
'''
NLLLoss的类定义如下所示：
torch.nn.NLLLoss(
     weight=None,
     ignore_index=-100,
     reduction="mean",
)
'''

import torch
import torch.nn as nn

model = nn.Sequential(
      nn.Linear(10, 3),
      nn.LogSoftmax()
)

criterion = nn.NLLLoss()
x = torch.randn(16, 10)
y = torch.randint(0, 3, size=(16,)) # (16, )
out = model(x) # (16, 3)
loss = criterion(out, y)
```

## 三、nn.**NLLLoss**

可参见nn.LogSoftmax下的说明，通常和LogSoftmax()结合一起使用

## 四、nn.CrossEntropyLoss

nn.CrossEntropyLoss用于计算 **多分类交叉熵损失，** 计算公式如下， ![[公式]](https://www.zhihu.com/equation?tex=z%3D%5Bz_0%2Cz_1%2C...%2Cz_%7BC-1%7D%5D) 表示一个样本的 **非softmax输出(即logits输出)** ，c表示该样本的标签，则损失函数公式描述如下：

![[公式]](https://www.zhihu.com/equation?tex=loss%28z%2Cc%29%3D-w+%5Ccdot+log%28%5Cfrac%7Be%5E%7Bz%5Bc%5D%7D%7D%7B%5Csum_%7Bj%3D0%7D%5E%7BC-1%7De%5E%7Bz%5Bj%5D%7D%7D%29%5C%5C+w+%3D+weight%5Bc%5D+%5Ccdot+1%5C%7Bc%5Cne++ignore%5C_index+%5C%7D)

由上可知，nn.CrossEntropyLoss是在一个类中**组合了nn.LogSoftmax()和nn.NLLLoss()**

```text
'''
类定义如下
torch.nn.CrossEntropyLoss(
    weight=None,
    ignore_index=-100,
    reduction="mean",
)
'''

import torch
import torch.nn as nn

model = nn.Linear(10, 3)
criterion = nn.CrossEntropyLoss()

x = torch.randn(16, 10)
y = torch.randint(0, 3, size=(16,)) # (16, )

logits = model(x) # (16, 3)
loss = criterion(logits, y)
```

五、torch.nn.Sigmoid 和 BCELoss()

Sigmoid多用于二分类或者多标签分类，Sigmoid的输出，每一维加起来不一定是和为1，也有可能输出的结果是 ![[公式]](https://www.zhihu.com/equation?tex=%5B0.1%EF%BC%8C0.5%EF%BC%8C0.7%5D) 这样子，所以这就可以进行多标签分类了，其公式如下： ![[公式]](https://www.zhihu.com/equation?tex=sigmoid%28z_i%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7Bz_i%7D%7D%5C%5C)

```python
## 代码如下
y = torch.rand(size=[2,2])

##  输出
>> tensor([[0.7696, 0.5311],
           [0.8619, 0.8131]])

net = nn.Sigmoid() 
output = net(y)

## 输出
>> tensor([[0.6834, 0.6297],
           [0.7031, 0.6928]])
```

Sigmoid函数可以用于激活函数或者用于二分类、多标签，下面是Sigmoid和BCELoss()结合使用 。

对于二分类而言，其中BCELoss()的计算方式如下， **此时Sigmoid输出是1维的（见下面代码）** ：

用 ![[公式]](https://www.zhihu.com/equation?tex=N) 表示样本数量，![[公式]](https://www.zhihu.com/equation?tex=p_n)表示预测第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个样本为正例的 **概率（经过sigmoid处理）** ，![[公式]](https://www.zhihu.com/equation?tex=y_n)表示第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个样本的标签，

![[公式]](https://www.zhihu.com/equation?tex=loss%28p%2C+y%29+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5EN-w_n%28y_n+%2A+log%28+p_n+%29%2B+%281-y_n%29+%2A+log%281+-p_n%29%29%5C%5C++w+%3D+weight%5Bc%5D+%5Ccdot+1%5C%7Bc%5Cne++ignore%5C_index+%5C%7D)

对于多标签分类问题，对于一个样本来说，它的各个label的分数加起来不一定等于1（Sigmoid输出）， **BCELoss在每个类维度上求cross entropy loss然后加和求平均得到** ，这里就体现了多标签的思想。

```python
## 下面是 Sigmoid 二分类模型
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 1),   ## 二分类
    nn.Sigmoid()
)
criterion = nn.BCELoss()

x = torch.randn(16, 10)  # (16, 10)
y = torch.empty(16).random_(2)  # shape=(16, ) 其中每个元素值为0或1

out = model(x)  # (16, 1)
out = out.squeeze(dim=-1)  # (16, )

loss = criterion(out, y)


## 下面是 Sigmoid 多标签分类模型
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 3),    ## 多分类
    nn.Sigmoid()
)
criterion = nn.BCELoss()

x = torch.randn(16, 10)  # (16, 10)
y = torch.empty(16, 3).random_(2)  # shape=(16, 3) 其中每个元素值为0或1，比如某一个样本是[1,0,1]  

out = model(x)  # (16, 3)

loss = criterion(out, y)
```

## 六、BCEWithLogitsLoss()

下面直接使用BCEWithLogitsLoss()，这个类将Sigmoid()和BCELoss()整合起来，比 纯粹使用BCELoss()+Sigmoid()更数值稳定。计算公式如下，其中![[公式]](https://www.zhihu.com/equation?tex=z_n)表示预测第n个样本为正例的 **得分（没经过Sgmoid处理）** ，![[公式]](https://www.zhihu.com/equation?tex=y_n)表示第n个样本的标签，![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta)表示sigmoid函数，则：

![[公式]](https://www.zhihu.com/equation?tex=loss%28z%2C+y%29+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5EN-w_n%28y_n+%2A+log%28%5Cdelta%28+z_n%29+%29%2B+%281-y_n%29+%2A+log%281+-%5Cdelta%28z_n%29%29%29%5C%5C++w+%3D+weight%5Bc%5D+%5Ccdot+1%5C%7Bc%5Cne++ignore%5C_index+%5C%7D)

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.BCEWithLogitsLoss()

x = torch.randn(16, 10)
y = torch.empty(16).random_(2)  # (16, )

out = model(x)  # (16, 1)
out = out.squeeze(dim=-1)  # (16, )

loss = criterion(out, y)
```

## 七、nn.MultiLabelSoftMarginLoss

nn.MultiLabelSoftMarginLoss() 和 nn.BCEWithLogitsLoss()没啥区别，唯一的区别就是nn.MultiLabelSoftMarginLoss()没有weighted参数

![[公式]](https://www.zhihu.com/equation?tex=loss%28z%2C+y%29+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bn%3D1%7D%5EN-w_n%28y_n+%2A+log%28%5Cdelta%28+z_n%29+%29%2B+%281-y_n%29+%2A+log%281+-%5Cdelta%28z_n%29%29%29%5C%5C+)

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.MultiLabelSoftMarginLoss()

x = torch.randn(16, 10)
y = torch.empty(16).random_(2)  # (16, )

out = model(x)  # (16, 1)
out = out.squeeze(dim=-1)  # (16, )

loss = criterion(out, y)
```

---

**2021.7.6 追更...**

**参考文献：**

[从理论到实践解决文本分类中的样本不均衡问题-技术圈**jishuin.proginn.com/p/763bfbd4c9be**![](https://pic2.zhimg.com/v2-8e1959705492c225e96bcd7e3d0f5929_ipico.jpg)](https://link.zhihu.com/?target=https%3A//jishuin.proginn.com/p/763bfbd4c9be)

## 从模型层面解决样本不平衡问题

本节主要从模型层面解决样本不均衡的问题。相比于控制正负样本的比例，我们还可以通过控制Loss损失函数来解决样本不均衡的问题。拿二分类任务来举例，通常使用交叉熵来计算损失，下面是交叉熵的公式：

![](https://pic3.zhimg.com/80/v2-50350e75251c8e1743645daf5e82ce76_1440w.jpg)

上面的公式中y是样本的标签，p是样本预测为正例的概率。

**1、类别加权Loss**

为了解决样本不均衡的问题，最简单的是基于类别的加权Loss，具体公式如下：

![](https://pic2.zhimg.com/80/v2-cee04f5cc8aeb74ccd745dc2fda47c65_1440w.jpg)

基于类别加权的Loss其实就是添加了一个参数a，这个a主要用来控制正负样本对Loss带来不同的缩放效果，一般和样本数量成反比。还拿上面的例子举例，有100条正样本和1W条负样本，那么我们设置a的值为10000/10100，那么正样本对Loss的贡献值会乘以一个系数10000/10100，而负样本对Loss的贡献值则会乘以一个比较小的系数100/10100，这样相当于控制模型更加关注正样本对损失函数的影响。通过这种基于类别的加权的方式可以从不同类别的样本数量角度来控制Loss值，从而一定程度上解决了样本不均衡的问题。

**2、Focal Loss**

上面基于类别加权Loss虽然在一定程度上解决了样本不均衡的问题，但是实际的情况是不仅样本不均衡会影响Loss，而且**样本的难易区分程度**也会影响Loss。下面是Focal Loss的计算公式：

![](https://pic1.zhimg.com/80/v2-5ee0c9eb4030f6999d135a2efa18dd8c_1440w.jpg)

相比于公式2来说，Focal Loss添加了参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 从置信的角度来加权Loss值。假如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 设置为0，那么公式3蜕变成了基于类别的加权也就是公式2；下面重点看看如何通过设置参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 来使得简单和困难样本对Loss的影响。当 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 设置为2时，对于模型预测为正例的样本也就是 ![[公式]](https://www.zhihu.com/equation?tex=p%3E0.5) 的样本来说，如果样本越容易区分那么 ![[公式]](https://www.zhihu.com/equation?tex=1-p) 的部分就会越小，相当于乘了一个系数很小的值使得Loss被缩小，也就是说对于那些比较容易区分的样本Loss会被抑制，同理对于那些比较难区分的样本Loss会被放大，这就是Focal Loss的核心： **通过一个合适的函数来度量简单样本和困难样本对总的损失函数的贡献。** 关于参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 的设置问题，Focal Loss的作者建议设置为2。

下面是一个Focal Loss的实现，感兴趣的小伙伴可以试试，看能不能对下游任务有积极效果

![](https://pic4.zhimg.com/80/v2-a76f8152b9c48fa6730f0b4f0789ff27_1440w.jpg)

**3、GHM Loss**

Focal Loss主要结合样本的难易区分程度来解决样本不均衡的问题，使得整个Loss的曲线平滑稳定的下降，但是对于一些特别难区分的样本比如离群点会存在问题。可能一个模型已经收敛训练的很好了，但是 **因为一些比如标注错误的离群点使得模型去关注这些样本，反而降低了模型的效果** 。比如下面的离群点图：

![](https://pic4.zhimg.com/80/v2-5178da4b6e63af0bec37032c233cf323_1440w.jpg)

针对Focal Loss存在的问题，2019年论文《Gradient Harmonized Single-stage Detector》中提出了GHM(gradient harmonizing mechanism) Loss。相比于Focal Loss从**置信度的角度**去调整Loss，GHM Loss则是从一定范围置信度p的样本数量(论文中称为梯度密度)去调整Loss。理解GHM Loss的第一步是先理解梯度模长的概念，梯度模长 ![[公式]](https://www.zhihu.com/equation?tex=g) 的计算公式如下：

![](https://pic4.zhimg.com/80/v2-49d839860c76c5d3a39ee27fa9d33d8b_1440w.jpg)

公式4中 ![[公式]](https://www.zhihu.com/equation?tex=p) 代表模型预测为1的概率值， ![[公式]](https://www.zhihu.com/equation?tex=p%5E%2A) 是标签值。也就是说如果样本越难区分，那么 ![[公式]](https://www.zhihu.com/equation?tex=g) 的值就越大。下面看看梯度模长 ![[公式]](https://www.zhihu.com/equation?tex=g) 和样本数量的关系图：

![](https://pic4.zhimg.com/80/v2-b7d4eac2dbc92363b0aa849efc9cc5a7_1440w.jpg)

从上图中可以看出样本中有很大一部分是容易区分的样本，也就是梯度模长 ![[公式]](https://www.zhihu.com/equation?tex=g) 趋于0的部分。但是还存在一些十分困难区分的样本，也就是上图中右边红圈中的样本。GHM Loss认为不仅仅要多关注容易区分的样本，这点和Focal Loss一致，同时还认为需要关注那些十分困难区分的样本，因为这部分样本可能是标注错误的离群点，过多的关注这部分样本不仅不会提升模型的效果，反而还会有一定的逆向效果。那么问题来了，怎么**同时抑制容易区分的样本和十分困难区分的样本**呢？

针对这个问题， **从上图中可以发现容易区分的样本和十分困难区分的样本都存在一个共同点：数量多** 。那么只要我们抑制一定梯度范围内数量多的样本就可以达到这个效果，GHM Loss通过梯度密度 ![[公式]](https://www.zhihu.com/equation?tex=GD%28g%29) 来表示一定梯度范围内的样本数量。这个其实有点像物理学中的密度，一定体积的物体的质量。梯度密度![[公式]](https://www.zhihu.com/equation?tex=GD%28g%29)的公式如下：

![](https://pic2.zhimg.com/80/v2-69f97cb85fb0f327ee922e5b4d80b8d9_1440w.jpg)

公式5中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta_%5Cepsilon%28g_k%2Cg%29) 代表样本中梯度模长 ![[公式]](https://www.zhihu.com/equation?tex=g) 分布在 ![[公式]](https://www.zhihu.com/equation?tex=%28g-%5Cepsilon%2F2%2C+g%2B%5Cepsilon%2F2%29) 范围里面的样本的个数，代表了 ![[公式]](https://www.zhihu.com/equation?tex=%28g-%5Cepsilon%2F2%2Cg%2B%5Cepsilon%2F2+%29) 区间的长度。公式里面的细节小伙伴们可以去论文里面详细了解。说完了梯度密度GD(g)的计算公式，下面就是GHM Loss的计算公式：

![](https://pic3.zhimg.com/80/v2-b8c20b25876ca557f84ac19ef7001eb6_1440w.jpg)

公式6中的Lce其实就是交叉熵损失函数，也就是公式1。下面是复现了GHM Loss的一个github上工程，有兴趣的小伙伴可以试试：[https://**github.com/libuyu/GHM_D**etection](https://link.zhihu.com/?target=https%3A//github.com/libuyu/GHM_Detection)
