RandomForestClassifier

参数

n_estimators : 随机森林中树的个数，即学习器的个数。
max_features : 划分叶子节点，选择的最大特征数目
n_features：在寻找最佳分割时要考虑的特征数量

max_depth : 树的最大深度，如果选择default=None，树就一致扩展，直到所有的叶子节点都是同一类样本，或者达到最小样本划分（min_samples_split）的数目。

min_samples_split : 最小样本划分的数目，就是样本的数目少于等于这个值，就不能继续划分当前节点了

min_samples_leaf : 叶子节点最少样本数，如果某叶子节点数目小于这个值，就会和兄弟节点一起被剪枝。

min_weight_fraction_leaf：叶子节点最小的样本权重和

max_leaf_nodes: 最大叶子节点数，默认是”None”，即不限制最大的叶子节点数

min_impurity_split：节点划分的最小不纯度，是结束树增长的一个阈值，如果不纯度超过这个阈值，那么该节点就会继续划分，否则不划分，成为一个叶子节点。

min_impurity_decrease : 最小不纯度减少的阈值，如果对该节点进行划分，使得不纯度的减少大于等于这个值，那么该节点就会划分，否则，不划分。

bootstrap :自助采样，又放回的采样，大量采样的结果就是初始样本的63.2%作为训练集。默认选择自助采样法。

oob_score : bool (default=False)
out-of-bag estimate，包外估计；是否选用包外样本（即bootstrap采样剩下的36.8%的样本）作为验证集，对训练结果进行验证，默认不采用。

n_jobs : 并行使用的进程数，默认1个，如果设置为-1，该值为总的核数。

random_state ：随机状态，默认由np.numpy生成

verbose：显示输出的一些参数，默认不输出。

属性(Attribute)

estimators_ :在RandomForestClassifier中，指的是决策树分类器的集合。

classes_:单个类别输出问题或者多类别输出问题中的类别标签数组。

n_classes_:单个类别输出问题或者多类别输出问题中的类别标签的个数。

n_features_ :数据集的特征个数，整型。

n_outputs_ :输出的个数，整型

feature_importances_ :The feature importances (the higher, the more important the feature)特征的权重

oob_score_ ：Score of the training dataset obtained using an out-of-bag estimate

oob_decision_function_ ：Decision function computed with out-of-bag estimate on the training set.

方法：

apply(X):Apply trees in the forest to X, return leaf indices.将森林中的树应用于X，返回叶索引

desicion_path(X):Return the decision path in the forest

fit(X,Y):在数据集（X,Y）上训练模型。

get_parms():获取模型参数

predict(X):预测数据集X的结果。

predict_log_proba(X):预测数据集X的对数概率。

predict_proba(X):预测数据集X的概率值。

score(X,Y):输出数据集（X,Y）在模型上的准确率。
