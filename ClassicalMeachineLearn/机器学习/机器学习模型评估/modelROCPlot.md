# Machine learning简单绘制ROC曲线

> ROC曲线，又可以称之为接受者操作特征曲线(Receiver Operating Characteristic Curve)，ROC曲线下的面积，称为AUC(Area Under Cureve)，可以衡量评估二分类模型的分类好坏。

本文视图使用Python中的Matplotlib模块来进行简单的ROC曲线的画法：

## **准备工作**

```python
#查看matplotlib的版本
import matplotlib
import matplotlib.pyplot as plt
matplotlib.__version__
#output
'3.3.3'

#首先导入实例乳腺癌二分类数据集
from sklearn.datasets import load_breast_cancer

#训练集测试集划分
from skelarn.model_selection import train_test_split  

#导入plot_roc_curve,roc_curve和roc_auc_score模块
from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score

#导入三个不同的分类器:LogisticRegression,DecisionTree和KNN
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#train_test_split划分
cancer = load_breast_cancer()
cancer_X = cancer.data
cancer_y = cancer.target
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer_X,cancer_y)

#创建模型
lr_clf = LogisticRegression(solver='saga',max_iter=10000)
dt_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()

#训练模型
lr_clf.fit(cancer_X_train,cancer_y_train)
dt_clf.fit(cancer_X_train,cancer_y_train)
knn_clf.fit(cancer_X_train,cancer_y_train)
```

## **方法一**

最新的matplotlib版本自动封装了绘制ROC曲线的 **plot_roc_curve()** 方法，可以快速便捷地直接绘制出不同模型的ROC曲线。

```text
#创建画布
fig,ax = plt.subplots(figsize=(12,10))
lr_roc = plot_roc_curve(estimator=lr_clf, X=cancer_X_test, 
                        y=cancer_y_test, ax=ax, linewidth=1)
dt_roc = plot_roc_curve(estimator=dt_clf, X=cancer_X_test,
                        y=cancer_y_test, ax=ax, linewidth=1)
knn_roc = plot_roc_curve(estimator=knn_clf, X=cancer_X_test,
                        y=cancer_y_test, ax=ax, linewidth=1)
#注意:这里的ax一定要传给所创建画布的ax,否则三个模型的ROC曲线分别绘制三张图而不在一张图中
#更改图例字体大小
ax.legend(fontsize=12)

#显示绘制的ROC曲线
plt.show()
```

![](https://pic3.zhimg.com/80/v2-0e4e443d421014dc24b7d47994392dfe_1440w.jpg)

## **方法二**

方法一的绘制ROC曲线的画法只针对于matplotlib比较新的情况，方法一的画法快速省力。但当我们所使用的版本比较旧时，就只能使用方法二来进行绘制。

```text
#首先我们使用建立好的模型对测试集数据进行预测预测的概率
score_lr = lr_clf.predict_proba(cancer_X_test)[:,1]
score_dt = dt_clf.predict_proba(cancer_X_test)[:,1]
score_knn = knn_clf.predict_proba(cancer_X_test)[:,1]

#使用roc_curve方法得到三个模型的真正率TP,假正率FP和阈值threshold
fpr_lr,tpr_lr,thres_lr = roc_curve(cancer_y_test,score_lr,)
fpr_dt,tpr_dt,thres_dt = roc_curve(cancer_y_test,score_dt,)
fpr_knn,tpr_knn,thres_knn = roc_curve(cancer_y_test,score_knn,)

print("LogitReg的AUC为:",auc(fpr_lr,tpr_lr))
print("DecisionTree的AUC为:",auc(fpr_dt,tpr_dt))
print("kNN的AUC为:",auc(fpr_knn,tpr_knn))

#Out
LogitReg的AUC为: 0.9852366478506297
DecisionTree的AUC为: 0.9423577941815023
kNN的AUC为: 0.9666739036039949

#创建画布
fig,ax = plt.subplots(figsize=(10,8))

#自定义标签名称label=''
ax.plot(fpr_lr,tpr_lr,linewidth=2,
        label='Logistic Regression (AUC={})'.format(str(round(auc(fpr_lr,tpr_lr),3))))
ax.plot(fpr_dt,tpr_dt,linewidth=2,
        label='Decision Tree (AUC={})'.format(str(round(auc(fpr_dt,tpr_dt),3))))
ax.plot(fpr_knn,tpr_knn,linewidth=2,
        label='K Nearest Neibor (AUC={})'.format(str(round(auc(fpr_knn,tpr_knn),3))))

#绘制对角线
ax.plot([0,1],[0,1],linestyle='--',color='grey')

#调整字体大小
plt.legend(fontsize=12)

plt.show()
```

![](https://pic1.zhimg.com/80/v2-1f7694830f43450d8f1bf123f6c8325c_1440w.jpg)

可以看到，方法二所绘制的ROC曲线所需要调整的参数相比于方法一略微多一些。

 **方法二相比于方法一主要是需要** ：

(1)手动添加标签名字，用于判断相应的模型；

(2)需要手动添加对应模型的AUC值，可以使用format配套AUC函数来解决；

(3)需要手动添加X轴和Y轴的名称，可以使用ax.set_xlabel()和ax.set_ylabel()来解决；

(4)需要手动添加图例，使用ax.legend()来显示。而方法一种即使不手动设置legend()图例也会自动显示，但需要修改字体大小和其他参数就需要手动调整ax.legend()参数。

## 注意事项

使用roc_auc_score()计算AUC的时候，传入的第一个参数应该是预测的真实标签，第二个参数**应该是模型预测为“真(1)”的概率**而不 **是模型预测的“0-1标签”** 。如果传入后者，会造成比实际AUC值偏低的情况。

```text
#得到模型预测的0-1标签
y_pre_lr = lr_clf.predict(cancer_X_test)
y_pre_dt = dt_clf.predict(cancer_X_test)
y_pre_knn = knn_clf.predict(cancer_X_test)

#模型预测的“真”概率值
score_lr = lr_clf.predict_proba(cancer_X_test)[:,1]
score_dt = dt_clf.predict_proba(cancer_X_test)[:,1]
score_knn = knn_clf.predict_proba(cancer_X_test)[:,1]

#正确做法
print("LogitReg的AUC为:",roc_auc_score(cancer_y_test,score_lr))
print("DecisionTree的AUC为:",roc_auc_score(cancer_y_test,score_dt))
print("kNN的AUC为:",roc_auc_score(cancer_y_test,score_knn))
#output
LogitReg的AUC为: 0.9852366478506297
DecisionTree的AUC为: 0.9423577941815023
kNN的AUC为: 0.9666739036039949

#错误做法
print("LogitReg的AUC为:",roc_auc_score(cancer_y_test,y_pre_lr))
print("DecisionTree的AUC为:",roc_auc_score(cancer_y_test,y_pre_dt))
print("kNN的AUC为:",roc_auc_score(cancer_y_test,y_pre_knn))
#output
LogitReg的AUC为: 0.9383412939643941
DecisionTree的AUC为: 0.9423577941815023
kNN的AUC为: 0.9223838471558836
```

刚刚我们用第二种方法画ROC曲线的时候，使用auc中的TPR和FPR验证了实际的AUC值应该是正确做法中的结果。

## 总结

介绍了两种简单画ROC曲线的方法

* 方法一：plot_roc_curve()，配合实际模型与X、y数据绘制，简单直接;
* 方法二：需roc_curve()传出FPR和TPR，以及auc()配合绘制，灵活性强;

注意计算AUC需要传入预测为“真(1)”概率，而不是实际的标签。

参考文献：

[1] [机器学习基础（1）- ROC曲线理解](https://link.zhihu.com/?target=https%3A//www.jianshu.com/p/2ca96fce7e81)

[2] 周志华.机器学习.ROC曲线

[3] scikit-learn官方文档.roc_curve.[https://**scikit-learn.org/stable**/modules/model_evaluation.html#scoring-parameter](https://link.zhihu.com/?target=https%3A//scikit-learn.org/stable/modules/model_evaluation.html%23scoring-parameter)
