from sklearn.neighbors import KNeighborsClassifier as kNN
import numpy as np

class kNNClassifier(object):
    def __init__(self, filePath, kValue):
        self.filePath = filePath
        self.kValue = kValue
    
    def loadData(self):
        data = np.loadtxt(self.filePath, dtype=object)
        #print(data)
        lables = data[:, -1]
        trainSet = data[:, 0:-1]
        #print("labels:\n", lables)
        #print("trainSet:\n", trainSet)
        return trainSet, lables
    
    def classficate(self, testSet):
        trainSet, labels = self.loadData()
        classMode = kNN(algorithm='auto', n_neighbors=self.kValue)
        classMode.fit(trainSet, labels)
        result = classMode.predict(testSet)
        print("result:\n", result)
        return result

if __name__ == "__main__":
    path = "/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/k邻近算法/海伦约会/data.txt"
    testSet = np.array([[42666, 13.2769, 0.540]])
    class1 = kNNClassifier(path, 3)
    class1.classficate(testSet)
    

        
