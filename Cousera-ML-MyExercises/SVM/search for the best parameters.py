from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('data/ex6data3.mat')
#print(mat.keys())
#print(mat)

training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')
cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')
#print(training.shape)

#visualize data
plt.scatter(training['X1'], training['X2'], s=50, c=training['y'], cmap='Reds')
plt.show()

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
combination = [(C, gamma) for C in candidate for gamma in candidate]

search = []
for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))

best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

print(best_score, best_param)

best_svc = svm.SVC(C=100, gamma=0.3)
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])

print(metrics.classification_report(cv['y'], ypred))

parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(training[['X1', 'X2']], training['y'])
print(clf.best_score_, clf.best_params_)
ypred_GS = clf.predict(cv[['X1', 'X2']])
print(metrics.classification_report(cv['y'], ypred_GS))