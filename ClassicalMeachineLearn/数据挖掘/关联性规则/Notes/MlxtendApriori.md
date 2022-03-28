# mlxtend手册（一）-频繁项目集

> Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.

mlxtend是一个拓展库，用来
mlxtend官网地址：[Welcome to mlxtend&#39;s documentation](https://links.jianshu.com/go?to=https%3A%2F%2Frasbt.github.io%2Fmlxtend%2F)

这次我使用这个主要是用来搞购物篮分析，也就是关联分析，下面我们就来开始入门了

##### 安装

这个之前写了一个，直接参考吧：[anaconda安装mlxtend](https://www.jianshu.com/p/b60c6409102c)

安装中没什么问题，只是用命令搞一下就好。

##### 频繁项目集

关于这一块儿，官网上有很详细的介绍，可以参考这里：[Frequent Itemsets via Apriori Algorithm](https://links.jianshu.com/go?to=https%3A%2F%2Frasbt.github.io%2Fmlxtend%2Fuser_guide%2Ffrequent_patterns%2Fapriori%2F)

频繁项目集，就是我们要在一堆订单中找到经常在一起被购买的商品组合。这也是我们关心的，哪些商品我可以哪来做组合销售，打包售卖，除了主观的判断，从数据上，我们的用户到底喜欢购买哪些？

官方例子中给到的样例数据如下：

这就相当于5笔订单，每笔订单中的商品。
一开始，我在这里产生了一个疑问，就是，比如说，同样的商品A、商品B，这样的订单如果有10笔，我该怎样提现出来呢？
我主观的以为之类忽略了订单的数量，只保留了出现过的商品，为此，我困惑了很久，后来在一篇文章中想明白了，后面也会分享。

后面，我们要调用mlxtend提供的方法，也就是传入参数，传入参数需要符合人家的要求，也就是接口规范，我们需要用到

这里，主要使用了TransactionEncoder的两个方法，一个fit，一个transform
先fit之后，mlxtend就可以知道有多少个唯一值，有了所有的唯一值，就可以转换成one-hot code了

![](https://upload-images.jianshu.io/upload_images/76024-0ed4cbd3274cc337.png?imageMogr2/auto-orient/strip|imageView2/2/w/580/format/webp)

如果要转为0,1形式，也非常方便

![](https://upload-images.jianshu.io/upload_images/76024-16d13d36e382ca9c.png?imageMogr2/auto-orient/strip|imageView2/2/w/359/format/webp)

上面说到的fit函数，它会初始化一个变量“columns_”

![](https://upload-images.jianshu.io/upload_images/76024-f14d8727b721f43d.png?imageMogr2/auto-orient/strip|imageView2/2/w/170/format/webp)

最后一步，把数据转换为DataFrame

![](https://upload-images.jianshu.io/upload_images/76024-4e1ecab9984c0a4c.png?imageMogr2/auto-orient/strip|imageView2/2/w/629/format/webp)

哦，下面才是最后1步，调用数据，输出频繁项目集

![](https://upload-images.jianshu.io/upload_images/76024-a170a703d372ff12.png?imageMogr2/auto-orient/strip|imageView2/2/w/358/format/webp)

这里输出的是支持度，支持度越高，说明出现的频率越高
这里的项目集显示不太友好，加个参数就好。

![](https://upload-images.jianshu.io/upload_images/76024-6666be23600d70d1.png?imageMogr2/auto-orient/strip|imageView2/2/w/354/format/webp)

上面，还有一个 min_support 参数，这是对支持度进行过滤，最小支持度为0.6

频繁项目集，就是如此，一步数据处理，一步接口调用，完事儿。

---

上面的数据已经出来了，我们排个序看看，顺便人工验证一下

![](https://upload-images.jianshu.io/upload_images/76024-16b8ee33a50aa372.png?imageMogr2/auto-orient/strip|imageView2/2/w/286/format/webp)

这个Kidney Beans支持度是1.0，也就是100%，说明所有的订单中都包含了这个商品，看一眼，的确是这样的
我们再看一个组合商品的，组合的可能才是我们更关注的
（Eggs，Kidney Beans），支持度0.8，说明有80%的订单都同时购买了这两款商品，再看一眼数据，一共5笔订单，包含这两款商品的订单有4笔，恩就是80%

这里会出现一个问题，就是刚刚提到的，我们想要看组合商品，单个商品的暂时不看，那就需要增加一个字段

![](https://upload-images.jianshu.io/upload_images/76024-e67b8d6abecef431.png?imageMogr2/auto-orient/strip|imageView2/2/w/618/format/webp)

然后，我们只要过滤下就好

![](https://upload-images.jianshu.io/upload_images/76024-67d67c13b5e3eeec.png?imageMogr2/auto-orient/strip|imageView2/2/w/374/format/webp)

54人点赞

[Python](https://www.jianshu.com/nb/17593325)
