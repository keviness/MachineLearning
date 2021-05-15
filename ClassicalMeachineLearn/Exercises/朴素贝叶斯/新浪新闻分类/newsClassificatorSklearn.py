import numpy as np
from sklearn.naive_bayes import MultinomialNB
import jieba as jb
import os
import random

class LoadDataFromFiles(object):
    def __init__(self, inputPath, filePostfix):
        self.inputPath = inputPath
        self.filePostfix = filePostfix

    def readFile(self, fileName):
        with open(fileName, 'r', encoding='utf-8') as f:
            content = f.read()
            f.close()
        #contentList = re.split(r"\W+", content)
        contentList = jb.cut(content, cut_all=False)
        #contentList = [tok.lower() for tok in contentList if len(tok) > 2]
        #print("content list:\n", contentList)

        return list(contentList)

    def loadFileFrom(self, subdirName, newsType):
        wordsArray = []
        for file in os.listdir(subdirName):
            filePath = os.path.join(subdirName, file)
            #print("file:\n", filePath)
            postfix = file.split('.')[-1]
            if postfix != self.filePostfix:
                print("Not find the %s file in the directory!" % fileName)
                continue
            contendList = self.readFile(filePath)
            wordsArray.append([newsType, contendList])
        #wordsArray = np.array(wordsArray, dtype='object')
        #print("words Array:\n", wordsArray)
        return wordsArray

    def loopInputPath(self):
        allDataList = []
        fileFolders = os.listdir(self.inputPath)
        for fileFolder in fileFolders:
            subdirName = os.path.join(self.inputPath, fileFolder)
            #print("subfile name:\n", subFileName)
            DataList = self.loadFileFrom(subdirName, fileFolder)
            allDataList.append(DataList)
        allDataList = np.array(allDataList, dtype=object)
        labelList = allDataList[:, :, 0]
        #trainWordsList = allDataList[:,:,1]
        #print("all data List:\n", allDataList)
        print("label list:\n", labelList)
        #print("trainWordsList:\n", trainWordsList)
        return allDataList





if __name__ == "__main__":
    inputPath = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/朴素贝叶斯/新浪新闻分类/Sample/'
    postfix = 'txt'
    instance1 = LoadDataFromFiles(inputPath, postfix)
    instance1.loopInputPath()

