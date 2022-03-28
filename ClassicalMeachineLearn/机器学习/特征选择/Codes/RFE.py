from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
iris = load_iris(as_frame=True)
irisData = iris.data
irisTarget = iris.target
print('irisData:\n', irisData)
print('irisTarget:\n', irisTarget)

model = RandomForestClassifier(n_estimators=700)

rfe = RFE(model, 3)
RFE_X_Train = rfe.fit_transform(irisData,irisTarget)
featureIndex = rfe.get_support(indices=True)
print('featureIndex:\n', featureIndex)
featureScore = rfe.score(irisData, irisTarget)
print('featureScore:\n', featureScore)

trainSet, testSet, trainLabel, testLabel = train_test_split(irisData, irisTarget, test_size=0.3)
model.fit(trainSet, trainLabel)
predictionforest = model.predict(testSet)
print("classification_report:\n", classification_report(testLabel, predictionforest))

'''
X, y = iris.data, iris.target  #iris数据集

#选择K个最好的特征，返回选择特征后的数据
features = SelectKBest(chi2, k=2)
print('features:\n', features)
X_new = features.fit_transform(X, y)
print('X:\n', X)
print('X_new:\n', X_new)
'''