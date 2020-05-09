import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *

path = 'ex5data1.mat'
X, y, Xval, yval, Xtest, ytest = load_data(path)
#print(X.shape, y.shape,Xval.shape,yval.shape,Xtest.shape,ytest.shape)

""" 
添加多项式前的学习曲线
X, Xval, Xtest = [np.insert(x, 0, np.ones(x.shape[0]),axis=1) for x in (X, Xval, Xtest)]
learning_curve(X, y, Xval, yval)
"""


X, Xval, Xtest = prepare_poly_data(X, Xval, Xtest, power=5)
X, Xval, Xtest = [np.insert(x, 0, np.ones(x.shape[0]),axis=1) for x in (X, Xval, Xtest)]
#print(X.shape,Xval.shape,Xtest.shape)

training_cost, cv_cost = [], []
reg_set = [0, 0.001, 0.003, 0.1, 0.3, 1, 3, 10]

for regRate in reg_set:
    fmin = linear_regression(X, y, regRate)

    tc = cost(fmin.x, X, y)
    cv = cost(fmin.x, Xval, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(reg_set, training_cost, label='Traning')
plt.plot(reg_set, cv_cost, label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()

print(reg_set[np.argmin(cv_cost)])

for regRate in reg_set:
    theta = linear_regression(X, y, regRate).x
    print('test cost(regRate={})={}'.format(regRate, cost(theta, Xtest, ytest)))