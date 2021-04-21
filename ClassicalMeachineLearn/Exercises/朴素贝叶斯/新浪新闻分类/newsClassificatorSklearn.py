import numpy as np
from sklearn.naive_bayes import MultinomialNB
import jieba as jb


class LoadDataFromFiles(object):
    def __init__(self, inputPath, filePostfix):
        self.inputPath = inputPath
        self.filePostfix = filePostfix

    def readFile(self, fileName):
        with open(fileName, 'r', encoding='unicode_escape') as f:
            content = f.read()
            f.close()
        #contentList = re.split(r"\W+", content)
        contentList = jb.cut(content, cut_all=False)
        #contentList = [tok.lower() for tok in contentList if len(tok) > 2]
        #print("content list:\n", contentList)

        return np.array(contentList)

    def loadFileFromDir(self):
        wordsArray = []
        for file in listdir(self.inputPath)[::-1]:
            fileName = self.inputPath+file
            #print(fileName)
            postfix = file.split('.')[-1]
            if postfix != self.filePostfix:
                print("Not find the %s file in the directory!" % fileName)
                continue
            contendList = self.readFile(fileName)
            wordsArray.append(contendList)
        wordsArray = np.array(wordsArray, dtype='object')
        #print("words Array:\n", wordsArray)
        return wordsArray
    

'''
class newsClassificator(object):
    def __init__(self, trainSet, labels):
        self.trainSet = trainSet
        self.labels = labels

    def 
'''

if __name__ == "__main__":

