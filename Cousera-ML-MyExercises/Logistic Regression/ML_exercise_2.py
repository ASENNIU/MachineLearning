import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

path = 'data/ex2data2.txt'
data = pd.read_csv(path,header=None,names=['Test 1','Test 2','Accept'])
#print(data.head(5))

"""
positive = data[data['Accept'].isin([1])]
negative = data[data['Accept'].isin([0])]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accept')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
"""

degree = 5
x1 = data['Test 1']
x2 = data['Test 2']
data.insert(3,'Ones',1)

for i in range(1,degree):
    for j in range(0,i):
        data['F'+str(i)+str(j)] = np.power(x1,i-j) * np.power(x2,j)

data.drop('Test 1',axis=1,inplace=True)
data.drop('Test 2',axis=1,inplace=True)
#print(data.head(5))

cols = data.shape[1]
X = data.iloc[:,1:cols]
y = data.iloc[:,0:1]
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.random.randn(X.shape[1])).T

lr = 0.02
_theta = theta.copy()
cost_list = []
for i in range(1000):
    cost_list.append(costReg(_theta,X,y,lr))
    _theta = _theta - lr *gradientReg(_theta,X,y,lr)

x = [ i for i in range(len(cost_list))]
plt.plot(x,cost_list)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()