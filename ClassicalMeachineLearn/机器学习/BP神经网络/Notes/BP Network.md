> 🔗 原文链接： [https://blog.csdn.net/weixin_423482...](https://blog.csdn.net/weixin_42348202/article/details/100568469)

### python调用 [sklearn ](https://so.csdn.net/so/search?q=sklearn&spm=1001.2101.3001.7020)库BP机器学习基于小样本进行痘痘预测尝试

* 背景：
* MLPClassifier() BP
* 处理过程：

  * 数据集
  * 证明下痘痘数据的真实性（自己每天记录），还是有点正态分布特征：
  * 数据标准化：
    * Excel标准化：
    * python的StandardScaler()标准化：
  * 代码时刻：
  * 运行结果：
* 结论：

# 背景：

# MLPClassifier() BP

这个暑假有幸接触到 [Anaconda ](https://so.csdn.net/so/search?q=Anaconda&spm=1001.2101.3001.7020)，甚至不知道具体怎么念，自己慢慢瞎摸乱搜慢慢学嘛。这个sklearn库，有点厉害。只需调用通过参数，解放了双手去编算法，得以零基础用机器学习。
本渣渣来到弗兰脸上容易长痘。自己平常喜欢将各种生活状态量化表示。遂有了自己的一个小excel。这个 [数据集 ](https://so.csdn.net/so/search?q=%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1001.2101.3001.7020)100多天。
于是，我决定对尝试下跨学科来进行医疗与机器学习。自己做的没什么价值  ~的，但万一言者无心，听者有意呢 ~ 。

# 处理过程：

## 数据集

尽管数据少，而且我只用了辣椒和油炸两个因素。我尝试加进去起床时间、天气，严重拉低预测率，遂放弃。但睡眠时间也是稍微拉低一点。
首先展示部分样本，我 **只用 **了油炸、辣椒两个参数。（本人数据 链接:https://pan.baidu.com/s/1kepl3NJm26IKbFOVSYdzqw 提取码:2h45）

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NzJkNGZlNWRlOWZkMmY0NjIzNDlmOTlhOWNjZDhhMjNfOHc2SWk3N2FKQTRBQ2V2WmhibjVyRlJPWGlRdWhrdWFfVG9rZW46Ym94Y250V3hMbGkyVlZKYzNGOUREZHI0Y1hBXzE2NjY2OTMzMDc6MTY2NjY5NjkwN19WNA)

## 证明下痘痘数据的真实性（自己每天记录），还是有点 [正态分布 ](https://so.csdn.net/so/search?q=%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83&spm=1001.2101.3001.7020)特征：

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NDJkNmVmYzlmOWM1OTE1MjY4NDk4ZmRmNmM5MTM0MDVfMUVQYzZReWpIQ2J6UkxjYnFvckoyaENYTFBNVXB1NnlfVG9rZW46Ym94Y25DanlIME5pOUk5YUc2UDg0U0FEbmdkXzE2NjY2OTMzMDc6MTY2NjY5NjkwN19WNA)

我通过这个直方图将痘痘分为2，3，4，5。4个等级

## 数据标准化：

### Excel标准化：

痘痘评分按上直方图进行量化后
睡眠时间先转浮点再通过excel表的if else if else…语句统一标准化

### python的StandardScaler()标准化：

再次统一对训练数据进行标准化操作

## 代码时刻：

```Python
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:10:07 2019
使用BP神经网络模型
@author: yiqing
"""
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.metrics import roc_curve
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#from sklearn.externals import joblib

def read():
    dates=pd.read_excel("D:\ziliao\jihua\日常记录3月.xlsx",sheet_name=1)
    print(dates.iloc[:98,16:19])#代表矩阵的16至19行 矩阵从（0，0）开始
    x_train=dates.iloc[30:79,17:19]
    print("xtrain:/n",x_train)
    y_train=dates.iloc[30:79,23]*10
    x_test=dates.iloc[80:129,17:19]
    y_test=dates.iloc[80:129,23]*10
    # 神经网络对数据尺度敏感，所以最好在训练前标准化，或者归一化，或者缩放到[-1,1]
    #数据标准化
    scaler = StandardScaler() # 标准化转换
    scaler.fit(x_test)  # 训练标准化对象
    x_test_Standard= scaler.transform(x_test)   # 转换数据集
    scaler.fit(x_train)  # 训练标准化对象
    x_train_Standard= scaler.transform(x_train)   # 转换数据集
    #
    bp=MLPClassifier(hidden_layer_sizes=(500, ), activation='relu', 
    solver='lbfgs', alpha=0.0001, batch_size='auto', 
    learning_rate='constant')
    bp.fit(x_train_Standard,y_train.astype('int'))
    y_predict=bp.predict(x_test_Standard)
  
    y_test1=y_test.tolist()
    y_predict=list(y_predict)
    #print(int(y_test1[1]))
    for i in range(len(y_test1)):
        y_test1[i]=int(y_test1[i])
      
    print('BP网络基于辣椒与油炸预测脸上痘痘评价报告：\n',classification_report(y_test.astype('int'),y_predict))
    print("真实数据：\t",y_test1)
    print("预测数据：\t",y_predict)

if __name__ == "__main__":
    read()
```

## 运行结果：

```Plaintext
BP网络基于辣椒与油炸预测脸上痘痘评价报告：
              precision    recall  f1-score   support

          2       0.00      0.00      0.00         0
          3       0.50      0.30      0.37        10
          4       0.85      0.89      0.87        38

avg / total       0.78      0.77      0.77        48

真实数据：    [4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3]
预测数据：    [4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 4]
```

# 结论：

（ 瞎扯的 ）
1）总预测率 78%还是不错的，主要原因大部分都是4,太稳定了，数据还是太少，想预测难都难
2）油炸与辣椒较影响痘痘，但相关性不是特别强。
3）go to bed时刻与总睡眠时长次之
4）天气、起床时间毫无关系
5) 我觉得应该痘痘与前好几天的饮食有关，正所谓冰冻一尺非一日之寒。
6) 个人认为：情绪状况、饮水、每日洗脸状况也可考虑在内。
7) 一方水土养一方人，回家不长痘，觉得这是在家喝粥，在学校不喝粥主稻造成也有点关系
8）用药。很有用。不过我没记录那几天
9）自己记录有失偏颇，无打分量准，完全看心情。
