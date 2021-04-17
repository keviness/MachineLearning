from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
from os import listdir
 
class NaiveBayesWordsFilter(object):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __crateVocabularyList(self):
        retSet = set([])
        for document in self.data:
            retSet = set(document) | retSet
        return list(retSet)

    def __wordToVector(self, inputWordSet):
        vocabularySet = self.__crateVocabularyList()
        result_list = np.zeros(len(vocabularySet), dtype=int)
        for word in inputWordSet:
            if word in vocabularySet:
                result_list[vocabularySet.index(word)] = 1
            else:
                print("%s isnot in the vocabuly!" % word)
        return result_list

    def classifyNaiveBayes(self, testWordSet):
        trainSet = np.array([self.__wordToVector(x) for x in self.data])
        testSet = np.array([self.__wordToVector(testWordSet)])
        #print("trainSet:\n", trainSet)
        #print("testSet:\n", testSet)

        classifyModle = MultinomialNB()
        classifyModle.fit(trainSet, self.labels)
        result = classifyModle.predict(testSet)
        print("result:\n", result)

        return result

class LoadDataFromFiles(object):
    def __init__(self, inputPath, filePostfix):
        self.inputPath = inputPath
        self.filePostfix = filePostfix
    
    def readFile(self, fileName):
        f = open(fileName, 'r', encoding='unicode_escape')
        content = f.read()
        f.close()
        contentList = re.split(r"\W+", content)
        contentList = [tok.lower() for tok in contentList if len(tok) > 2]
        #print("content list:\n", contentList)

        return np.array(contentList)

    def loadFileFromDir(self):
        wordsArray = []
        for file in listdir(self.inputPath)[::-1]:
            fileName = self.inputPath+file
            #print(fileName)
            postfix = file.split('.')[-1]
            if postfix != self.filePostfix:
                print("Not find the %s file in the directory!"%fileName )
                continue
            contendList = self.readFile(fileName)
            wordsArray.append(contendList)
        wordsArray = np.array(wordsArray, dtype='object')
        #print("words Array:\n", wordsArray)
        return wordsArray

if __name__ == '__main__':
    inputPath1 = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/朴素贝叶斯/垃圾邮件过滤器/email/ham/'
    inputPath2 = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/朴素贝叶斯/垃圾邮件过滤器/email/spam/'
    inputPath3 = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/朴素贝叶斯/垃圾邮件过滤器/email/test/'
    filePostfilx = 'txt'
    instance1 = LoadDataFromFiles(inputPath1, filePostfilx)
    wordsData1 = instance1.loadFileFromDir()
    #print("wordsData1:\n", wordsData1)
    instance2 = LoadDataFromFiles(inputPath2, filePostfilx)
    wordsData2 = instance2.loadFileFromDir()
    #print("wordsData2:\n", wordsData2)
    data = np.concatenate((wordsData1, wordsData2))
    labels = ['No spam']*25+['yes spam']*25
    #--test vector--
    testInstance1 = LoadDataFromFiles(inputPath3, filePostfilx)
    testWordsDatas = testInstance1.loadFileFromDir()
    for testWordsData in testWordsDatas:
        #print("data:\n", data)
        print('testWordsData:\n', testWordsData)
        classModel = NaiveBayesWordsFilter(data, labels)
        classModel.classifyNaiveBayes(testWordsData)
            