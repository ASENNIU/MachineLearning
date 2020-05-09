from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import scipy.io as sio

mat_train = sio.loadmat('data/spamTrain.mat')
#print(mat_tr.keys())

X_train, y_train = mat_train.get('X'), mat_train.get('y').ravel()
#print(X_train.shape, y_train.shape)

mat_test = sio.loadmat('data/spamTest.mat')
#print(mat_test.keys())
X_test, y_test = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
#print(X_test.shape, y_test.shape)

svc = svm.SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print("SVM: \n")
print(metrics.classification_report(y_test, pred))

logistic = LogisticRegression()
logistic.fit(X_train, y_train)
pred_lg = logistic.predict(X_test)
print("Logitic Regression:\n")
print(metrics.classification_report(y_test, pred_lg))