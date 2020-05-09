import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

path = 'data/ex2data1.txt'
data = pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])
#print(data.head(5))

data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.random.randn(X.shape[1])).T

_theta = theta.copy()
cost_list = []
for i in range(2000):
    cost_list.append(cost(_theta,X,y))
    _theta = _theta - 0.001 * gradient_descent(_theta,X, y)

x = [i for i in range(len(cost_list))]

fig, ax = plt.subplots(1,2,figsize=(16,8))
ax[0].plot(x,cost_list)
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel("Cost")


#创建散点图，并使用颜色编码来可视化
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
f = (0.5 - _theta[0,0] - positiva['Exam 1'] * _theta[1,0]) / _theta[2,0]
ax[1].scatter(positive['Exam 1'],positiva['Exam 2'],s=50,c='b',marker='o',label='Admitted')
ax[1].scatter(negative['Exam 1'],negative['Exam 2'],s=50,c='r',marker='x',label='No Admitted')
ax[1].plot(positive['Exam 1'],f)
ax[1].legend()
ax[1].set_xlabel('Exam 1 Score')
ax[1].set_ylabel('Exam 2 Score')
plt.show()

P = predict(_theta,X)
print(P)

