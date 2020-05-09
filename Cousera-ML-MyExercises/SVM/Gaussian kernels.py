import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import seaborn as sns
import scipy.io as sio
import numpy as np

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))

mat = sio.loadmat('data/ex6data2.mat')
#print(mat)
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')
#print(data.head(5))

#visualize data
sns.set(context="notebook", style='white', palette=sns.diverging_palette(240,10,n=2))
sns.lmplot('X1','X2',hue='y',data=data,size=5,fit_reg=False, scatter_kws={'s':10})
plt.show()

#try built-in Gaussian Kernel of sklearn
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])
print(svc.score(data[['X1', 'X2']], data['y']))

predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')
plt.show()

