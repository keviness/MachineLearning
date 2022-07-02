作者：德川Captain
链接：https://www.zhihu.com/question/271470776/answer/403799556
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

sklearn支持多类别（Multiclass）分类和多标签（Multilabel）分类：

* [多类别分类](https://www.zhihu.com/search?q=%E5%A4%9A%E7%B1%BB%E5%88%AB%E5%88%86%E7%B1%BB&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A403799556%7D)：超过两个类别的分类任务。多类别分类假设每个样本属于且仅属于一个标签，类如一个水果可以是苹果或者是桔子但是不能同时属于两者。
* [多标签分类](https://www.zhihu.com/search?q=%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A403799556%7D)：给每个样本分配一个或多个标签。例如一个新闻可以既属于体育类，也属于文娱类。

sklearn的[官方文档](https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/multiclass.html%23multiclass)给出了支持多标签分类的类，包括如下：

* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html%23sklearn.tree.DecisionTreeClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.tree.DecisionTreeClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html%23sklearn.tree.ExtraTreeClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.tree.ExtraTreeClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html%23sklearn.ensemble.ExtraTreesClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.ensemble.ExtraTreesClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html%23sklearn.neighbors.KNeighborsClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.neighbors.KNeighborsClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html%23sklearn.neural_network.MLPClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.neural_network.MLPClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html%23sklearn.neighbors.RadiusNeighborsClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.neighbors.RadiusNeighborsClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html%23sklearn.ensemble.RandomForestClassifier" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.ensemble.RandomForestClassifier</a>`
* `<a href="https://link.zhihu.com/?target=http%3A//scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html%23sklearn.linear_model.RidgeClassifierCV" class=" wrap external" target="_blank" rel="nofollow noreferrer" data-za-detail-view-id="1043">sklearn.linear_model.RidgeClassifierCV</a>`

以[决策树](https://www.zhihu.com/search?q=%E5%86%B3%E7%AD%96%E6%A0%91&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A403799556%7D)举例，给出如下实现过程

## 数据准备

```python
from sklearn.datasets import make_multilabel_classification

# Generate a random multilabel classification problem.
# For each sample, the generative process is:
#     pick the number of labels: n ~ Poisson(n_labels)
#     n times, choose a class c: c ~ Multinomial(theta)
#     pick the document length: k ~ Poisson(length)
k times, choose a word: w ~ Multinomial(theta_c)
X, Y = datasets.make_multilabel_classification(n_samples=10, n_features=5, n_classes=3, n_labels=2)
```

生成的X和Y为如下形式的数据：

  ![](https://pic2.zhimg.com/80/v2-f4daeaa3b7e233259ee9301aab0f3ec6_1440w.jpg?source=1940ef5c)

## 分类

```text
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Split dataset to 8:2
X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.2)

cls = DecisionTreeClassifier()
cls.fit(X_train, Y_train)
```

## 多标签分类评估

```text
from sklearn import metrics

Y_pred = cls.predict(X_test)
```

Y_test和Y_pred值如下：

  ![](https://pic1.zhimg.com/80/v2-a6d8e9334a78cac5544a8e9fd2fd9f0c_1440w.jpg?source=1940ef5c)

```text
metrics.f1_score(Y_test, Y_pred, average="macro")
# 0.666
metrics.f1_score(Y_test, Y_pred, average="micro")
# 0.8
metrics.f1_score(Y_test, Y_pred, average="weighted")
# 1.0
metrics.f1_score(Y_test, Y_pred, average="samples")
# 0.4
```

## [概率预测](https://www.zhihu.com/search?q=%E6%A6%82%E7%8E%87%E9%A2%84%E6%B5%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A403799556%7D)

```text
Y_prob = cls.predict_proba(X_test)
```

X_test和Y_prob值如下：

  ![](https://pic2.zhimg.com/80/v2-3291897ac9b73c96e801adfc453091d7_1440w.jpg?source=1940ef5c)

  ![](https://picx.zhimg.com/80/v2-6543e3bf30560f911725792791da6386_1440w.jpg?source=1940ef5c)

> `predict_proba`( *X* )
> X：array-like or sparse matrix of shape = [n_samples, n_features]
> RETURN：array of shape = [n_samples, n_classes], or a list of n_outputs

[编辑于 2018-06-02 10:42](//www.zhihu.com/question/271470776/answer/403799556)
