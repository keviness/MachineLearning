import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 

# ----二，读取文件及文件预处理----
path = "/Users/kevin/Desktop/program files/MeachineLearning/妇科用药推荐/Data Precess/Data/妇科用药.xlsx"
sheetName = "ExperimentData"
outputPath = '/Users/kevin/Desktop/program files/MeachineLearning/妇科用药推荐/Data Precess/Result/'

# ---load Data and prepare handle---
def getData(path):
    sourceDataFrame = pd.read_excel(path, sheet_name=sheetName)
    print("sourceDataFrame:\n", sourceDataFrame)
    sourceDataFrame.drop_duplicates(subset=['方剂名称'],keep='first', inplace=True)
    FunctionsArray = sourceDataFrame["方剂名称"].values
    HerbsArray = sourceDataFrame["标准药物名称"].values
    contentArray = sourceDataFrame["主治"].values
    #print("sentences:\n", contentArray)
    HerbsListArray = [e.split('、') for e in HerbsArray]
    #print("combListArray:\n", combListArray)
    herbsSetArray = list(set(sum(HerbsListArray,[])))
    #print("herbsSetArray:\n", herbsSetArray)
    return FunctionsArray, HerbsArray, herbsSetArray, HerbsListArray, contentArray

def getVector(vocabList, inputSetArray):
    resultArray = []
    for inputSet in inputSetArray:
        returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
        for word in inputSet:  # 遍历每个词条
            if word in vocabList:  # 如果词条存在于词汇表中，则置1
                returnVec[vocabList.index(word)] = 1
            else:
                print("the word: %s is not in my Vocabulary!" % word)
        resultArray.append(returnVec)
    resultArray = np.array(resultArray)
    #print('resultArray:\n', resultArray.shape)
    dataFrame = pd.DataFrame(data=resultArray, columns=vocabList)
    #print('dataFrame:\n', dataFrame)
    return dataFrame  # 返回文档向量
    
if __name__ == '__main__':
    FunctionsArray, HerbsArray, herbsSetArray, HerbsListArray, contentArray = getData(path)
    PropertyVectorDataFrame = getVector(herbsSetArray, HerbsListArray)
    PropertyVectorDataFrame.insert(0,'方剂名称', FunctionsArray)
    PropertyVectorDataFrame.insert(1,'中药组成', HerbsArray)
    PropertyVectorDataFrame.insert(2,'主治', contentArray)
    print('PropertyVectorDataFrame:\n', PropertyVectorDataFrame)
    PropertyVectorDataFrame.to_csv(outputPath+'FuNvVectorInfo'+'.csv', index=False)
    print('Write to excel file successfully!')