import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as ds# 数据准备
from  sklearn.neighbors import KNeighborsClassifier as kNN
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings("ignore")

iris = ds.load_iris(as_frame=True)
irisData = iris.data
irisTarget = iris.target.values
print('irisData:\n', irisData)
print('irisTarget:\n', irisTarget)

# 选择模型 
# RandomForest
'''
model = RandomForestClassifier()
# 参数搜索空间
param_grid = {
    'max_depth': np.arange(1, 20, 1),
    'n_estimators': np.arange(1, 50, 10),
    'max_leaf_nodes': np.arange(2, 100, 10)
}
'''
# kNN
model = kNN()
# 参数搜索空间
param_grid = {
    'n_neighbors':list(range(2,11)),
    'algorithm':['auto','ball_tree','kd_tree','brute']
    }
# 网格搜索模型参数
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_micro')
grid_search.fit(irisData, irisTarget)
print('best_params_:\n', grid_search.best_params_)
print('grid_search best_score_:\n', grid_search.best_score_)
print('best_estimator_:\n', type(grid_search.best_estimator_))

''''
# 随机搜索模型参数
rd_search = RandomizedSearchCV(model, param_grid, n_iter=200, cv=5, scoring='f1_micro')
rd_search.fit(x, y)
print(rd_search.best_params_)
print(rd_search.best_score_)
print(rd_search.best_estimator_)
'''

# 实例化贝叶斯搜索器,传入分类模型和参数组合以及迭代次数
Bayes = BayesSearchCV(model, param_grid, n_iter=15, random_state=14)
Bayes.fit(irisData,irisTarget)

# best parameter combination
print("Bayes最优参数:",Bayes.best_params_)

# Score achieved with best parameter combination
print("Bayes最优分数:",Bayes.best_score_)

# all combinations of hyperparameter
#print("交叉验证参数组合:",Bayes.cv_results_['params'])

# average scores of cross-validation
#print("交叉验证测试分数:",Bayes.cv_results_['mean_test_score'])
