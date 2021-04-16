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
        print("trainSet:\n", trainSet)
        print("testSet:\n", testSet)

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
        print("content list:\n", contentList)

        return np.array(contentList)

    def loadFileFromDir(self):
        wordsArray = []
        i=1
        for file in listdir(self.inputPath)[::-1]:
            fileName = self.inputPath+file
            print(fileName)
            postfix = file.split('.')[-1]
            if postfix != self.filePostfix:
                print("Not find the %s file in the directory!"%fileName )
                continue
            '''
            contendList = self.readFile(fileName)
            wordsArray.append(contendList)
            i+=1
        print("words Array:\n", wordsArray)

        return wordsArray
        '''

if __name__ == '__main__':
    inputPath = '/Users/kevin/Desktop/program files/MeachineLearning/ClassicalMeachineLearn/Exercises/朴素贝叶斯/垃圾邮件过滤器/email/ham/'
    filePostfilx = 'txt'
    instance = LoadDataFromFiles(inputPath, filePostfilx)
    instance.loadFileFromDir()
            

            



