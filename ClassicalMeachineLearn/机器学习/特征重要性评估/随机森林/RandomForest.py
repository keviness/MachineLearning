from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

outputPath = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/机器学习/特征重要性评估/随机森林/Data/'
inputPath = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/机器学习/特征重要性评估/随机森林/Data/ExperimentData.xlsx'

'''
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header = None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
              'Alcalinity of ash', 'Magnesium', 'Total phenols', 
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
df.to_excel(outputPath+'ExperimentData.xlsx', index=False)
'''
def getData(inputPath):
    df = pd.read_excel(inputPath)
    x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    trainSet, testSet, trainLabel, testLabel = train_test_split(x, y, test_size = 0.3, random_state = 0)
    feat_labels = df.columns[1:]
    return trainSet, testSet, trainLabel, testLabel, feat_labels

def RandomForest(trainSet, trainLabel, testSet):
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(trainSet, trainLabel)
    result = forest.predict(testSet)
    importance = forest.feature_importances_
    print('importance:\n', importance)
    return result, importance
    
def getAccuracy(predictresult, testLabel):
    acc = sum(predictresult==testLabel)/len(testLabel)
    print('accruacy:\n', acc)
    
if __name__ == '__main__':
    trainSet, testSet, trainLabel, testLabel, feat_labels = getData(inputPath)
    predictResult, importances = RandomForest(trainSet, trainLabel, testSet)
    print('testLabel:\n', testLabel)
    print('preResult:\n',predictResult)
    indices = np.argsort(importances)[::-1]
    #print('indices:\n', indices)
    
    for f in range(trainSet.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    getAccuracy(predictResult, testLabel)
    