from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# this will generate a random multi-label dataset
X, y = make_multilabel_classification(sparse = True, n_samples=200, n_labels=15,n_classes=10,n_features=15,return_indicator = 'sparse', allow_unlabeled = False)

print('X:\n', X.toarray().shape)
print('y:\n', y.toarray().shape)

'''
# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
x_train = X[:10,:]
y_train = y[:10,:]
classifier.fit(x_train, y_train)

# predict
x_test = X[10:,:]
y_test = y[10:,:]
predictions = classifier.predict(x_test)
print('y_test:\n', y_test.toarray())
print('predictions:\n', predictions.toarray())

accuracy = accuracy_score(y_test,predictions)
print('accuracy:\n', accuracy)
'''
'''
# using classifier chains
from skmultilearn.problem_transform import ClassifierChain

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier = ClassifierChain(GaussianNB())

# train
x_train = X[:10,:]
y_train = y[:10,:]
classifier.fit(x_train, y_train)

# predict
x_test = X[10:,:]
y_test = y[10:,:]
predictions = classifier.predict(x_test)
accuracy = accuracy_score(y_test,predictions)
print('accuracy:\n', accuracy)
'''
'''
# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
x_train = X[:10,:]
y_train = y[:10,:]
classifier.fit(x_train, y_train)

# predict
x_test = X[10:,:]
y_test = y[10:,:]
predictions = classifier.predict(x_test)

accuracy = accuracy_score(y_test,predictions)
print('accuracy:\n', accuracy)
'''

from skmultilearn.adapt import MLkNN
classifier = MLkNN(k=10)

# train
x_train = X[:100,:]
y_train = y[:100,:]
classifier.fit(x_train, y_train)

# predict
x_test = X[100:,:]
y_test = y[100:,:]
predictions = classifier.predict(x_test)

accuracy = accuracy_score(y_test,predictions)
print('accuracy:\n', accuracy)