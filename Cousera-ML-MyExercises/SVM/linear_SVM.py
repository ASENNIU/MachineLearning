import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('data/ex6data1.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
#print(data.head(5))


#visualize data
P = data[data['y'].isin([1])]
N = data[data['y'].isin([0])]
#ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds') #当c的参数是浮点数时可以像这样一样用cmap表示一系列不同颜色即颜色深度不一样
plt.scatter(P['X1'], P['X2'], s=50, c='b')
plt.scatter(N['X1'], N['X2'], s=50, c='r')
plt.title('Raw data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
print(svc1.score(data[['X1', 'X2']], data['y']))
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
print(svc100.score(data[['X1', 'X2']], data['y']))
data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(1,2,figsize=(18,8))
ax[0].scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax[0].set_title('SVM1 Confidence')

ax[1].scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
ax[1].set_title('SVM100 Confidence')
plt.show()

print(data.head(5))