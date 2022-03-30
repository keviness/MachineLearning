# 从skopt包中导入BayesSearchCV搜索器
# 安装scikit-optimize
# pip install scikit-optimize
from sklearn.neighbors import KNeighborsClassifier as kNN
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings("ignore")
import sklearn.datasets as ds# 数据准备

iris = ds.load_iris(as_frame=True)
irisData = iris.data
irisTarget = iris.target.values
print('irisData:\n', irisData)
print('irisTarget:\n', irisTarget)
# parameter ranges are specified by one of below
# from skopt.space import Real, Categorical, Integer

knn = kNN()
# 定义贝叶斯搜索的参数组合
grid_param = {'n_neighbors':list(range(2,11)),
              'algorithm':['auto','ball_tree','kd_tree','brute']}

# 实例化贝叶斯搜索器,传入分类模型和参数组合以及迭代次数
Bayes = BayesSearchCV(knn,grid_param,n_iter=10,random_state=14)
Bayes.fit(irisData,irisTarget)

# best parameter combination
print("最优参数:",Bayes.best_params_)

# Score achieved with best parameter combination
print("最优分数:",Bayes.best_score_)

# all combinations of hyperparameter
#print("交叉验证参数组合:",Bayes.cv_results_['params'])

# average scores of cross-validation
#print("交叉验证测试分数:",Bayes.cv_results_['mean_test_score'])
