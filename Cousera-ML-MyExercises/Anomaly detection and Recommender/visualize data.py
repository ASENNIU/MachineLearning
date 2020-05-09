import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns
sns.set(context='notebook', palette=sns.color_palette('RdBu'))
import pandas as pd
import numpy as np
from scipy import stats

mat = sio.loadmat('data/ex8data1.mat')
print(mat.keys())
X = mat.get('X')

#visualize training data
sns.regplot('Latency', 'Throughput', data=pd.DataFrame(X, columns=['Latency', 'Throughput']), fit_reg=False, scatter_kws={'s':20, 'alpha':0.5})
plt.show()

mu = X.mean(axis=0)
print(mu, '\n')

cov = np.cov(X.T)
print(cov)

#create multi_var Gaussian model
multi_normal = stats.multivariate_normal(mu, cov)

#create a grid
x, y = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((x, y))
print(pos.shape)

fig, ax = plt.subplots()

#plot probability density
ax.contourf(x, y, multi_normal.pdf(pos), cmap='Blues')

#plot original data points
sns.regplot('Latency', 'Throughput',data=pd.DataFrame(X, columns=['Latency', 'Throughput']), fit_reg=False, ax=ax, scatter_kws={"s":10, "alpha":0.4})
plt.show()

