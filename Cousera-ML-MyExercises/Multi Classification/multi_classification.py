import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from functions import *

data = loadmat('ex3data1.mat')
#print(data['X'].shape,data['y'].shape)

rows = data['X'].shape[0]
params = data['X'].shape[1]

"""查看数据的维度
X = np.insert(data['X'],0,values=np.ones(rows),axis=1)
theta = np.zeros(params + 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = y_0.reshape(rows,1)
print(X.shape,y_0.shape,theta.shape,all_theta.shape)
"""

print(np.unique(data['y']))

all_theta = one_vs_all(data['X'],data['y'],10,1)

y_pred = predict_all(data['X'],all_theta)
correct = [1 if a == b else 0 for (a,b) in zip(y_pred,data['y'])]
accuracy = sum(map(int,correct)) / float(len(correct))
print('accuracy = {0}%'.format(accuracy * 100))