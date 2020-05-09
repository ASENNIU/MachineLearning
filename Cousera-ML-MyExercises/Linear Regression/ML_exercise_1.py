import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

path = 'data/ex1.txt'
data = pd.read_csv(path, header=None, sep='\t', names=['Population', 'Profit'])
data.insert(0, 'Ones', 1)
"""展示数据
print(data.head(5))
print(data.describe())
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.show()
"""
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
# print(X.head(5))
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.random.randn(X.shape[1])).T

theta_train, cost = gradientDescent(X, y, theta)
x = np.linspace(X[:,1].min(),X[:,1].max(),100)
f = theta_train[0,0] + theta_train[1,0] * x
fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].plot(x,f,'r',label='Prediction')
ax[0].scatter(data.Population, data.Profit,label='Traning Data')
ax[0].legend()
ax[0].set_xlabel('Population')
ax[0].set_ylabel('Profit')
ax[0].set_title('Predicted Profit vs. Population Size')
ax[1].plot(range(1000),cost)
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Cost')
ax[1].set_title('Error vs. Train Epoch')
plt.show()


