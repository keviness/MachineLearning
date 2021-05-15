from sklearn import tree
import numpy as np

class DecisionTreeClassficator(object):
    def __init__(self, dataSet):
        self.dataSet = dataSet
        
    def Classficate(self, maxDepth, labels, testSet):
        classficateMode = tree.DecisionTreeClassifier(max_depth=maxDepth)
        classficateMode.fit(self.dataSet, labels)
        result = classficateMode.predict(testSet)
        print("result:\n", result)
        return result

if __name__ == "__main__":
    dataSet = np.array([[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']])

    testSet = np.array([[1, 1, 0, 2]])
    class1 = DecisionTreeClassficator(dataSet[:, 0:-1])
    class1.Classficate(maxDepth=4, labels=dataSet[:, -1], testSet=testSet)