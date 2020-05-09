import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(context='notebook', palette=sns.color_palette('RdBu'))
import pandas as pd
import numpy as np
from functions import *

mat = sio.loadmat('data/ex8data1.mat')
#print(mat.keys())
X = mat.get('X')

#divide original validation data into validation and test set
Xval, Xtest, yval, ytest = train_test_split(mat.get('Xval'), mat.get('yval'), test_size=0.5)

e, fs = selct_threshold(X, Xval, yval)
print('Best epsilon: {}\nBest F-score on validation data: {}'.format(e, fs))

multi_normal, y_pred = predict(X, Xval, e, Xtest, ytest)

data = pd.DataFrame(Xtest, columns=['Latency', 'Throughput'])
data['y_pred'] = y_pred

x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots()

ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

sns.regplot('Latency', 'Throughput', data=data, fit_reg=False, scatter_kws={'s':10, 'alpha':0.4})

#mark the predicted anamoly of CV data. We should have a test set for this...
anamoly_data = data[data['y_pred'] == 1]
ax.scatter(anamoly_data['Latency'], anamoly_data['Throughput'], marker='x', s=50)
plt.show()