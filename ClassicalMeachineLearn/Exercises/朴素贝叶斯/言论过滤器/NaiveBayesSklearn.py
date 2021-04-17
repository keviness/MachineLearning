from sklearn.naive_bayes import MultinomialNB
import numpy as np

class NaiveBayesWords(object):
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
                print("%s isnot in the vocabuly!" %word)
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

if __name__ == "__main__":
    data = np.array([np.array(['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']),  # 切分的词条
            np.array(['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']),
            np.array(['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']),
            np.array(['stop', 'posting', 'stupid', 'worthless', 'garbage']),
            np.array(['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']),
            np.array(['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'])], dtype = object)
    print('data:\n', data)
    labels = [0, 1, 0, 1, 0, 1]
    testWordSet1 = ['love', 'my', 'dalmation', 'dog', 'stupid', 'worthless']
    testWordSet2 = ['stupid', 'garbage']
    classInstance = NaiveBayesWords(data, labels)
    classInstance.classifyNaiveBayes(testWordSet1)

        
