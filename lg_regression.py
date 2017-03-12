import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy

trainX = numpy.load('data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('data/tinyY.npy') 

trainX, trainY = shuffle(trainX,trainY)

classif = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(trainX,trainY, test_size=0.3, random_state=42)

classif.fit(X_train.reshape((X_train.shape[0],-1)),y_train)
print classif.score(X_test.reshape((X_test.shape[0],-1)),y_test)

