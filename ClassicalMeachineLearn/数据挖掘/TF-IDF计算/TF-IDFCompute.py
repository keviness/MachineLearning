# coding:utf-8  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer  
#语料  
corpus = [  
    'This;is;the;first;document.',  
    'This;is;the;second;second;document.',  
]  
#将文本中的词语转换为词频矩阵  
vectorizer = CountVectorizer()  
#计算个词语出现的次数  
X = vectorizer.fit_transform(corpus)  
#获取词袋中所有文本关键词  
word = vectorizer.get_feature_names()  
print('word:\n',word)  
#查看词频结果  
print(X.toarray())

# ----------------------------------------------------

#类调用  
transformer = TfidfTransformer()  
#print(transformer)
#将词频矩阵X统计成TF-IDF值  
tfidf = transformer.fit_transform(X.toarray())  
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
print('tf-idf:\n',tfidf.toarray()) 
