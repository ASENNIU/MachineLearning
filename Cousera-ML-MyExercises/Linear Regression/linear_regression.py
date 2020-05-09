#coding: utf-8
#linear_regression
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    读取数据
    arg:
        filename文件名
    returns:
        x 训练样本集矩阵
        y 标签矩阵
    """
    numFeat = len(open(filename).readline().split('\t')) -1
    X = []
    y = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        X.append(lineArr)
        y.append(float(curLine[-1]))
    return np.array(X), np.array(y)

def lr_cost(theta, X, y):
    m = X.shape[0]  # m为样本数
    inner = X @ theta - y
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m

def batch_gradient_decent(theta ,X, y, epoch, alpha=0.02):
    cost_data = [lr_cost(theta, X,y)]
    _theta = theta.copy()
    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))
    return _theta, cost_data

def featureNormalize(X):
    """
    特征标准化处理
    Args:
        X 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    for j in range(n):
        features = X[:,j]
        meanval = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:,j] = (features - meanval)/std
        else:
            X[:, j] = 0
    return X


def normalEqn(X,y):
   inv =  np.linalg.inv(X.T @ X)
   return inv @ X.T @ y

X, y = loadDataSet('data/ex0.txt')
theta = np.random.randn(X.shape[1])
theta1 = normalEqn(X,y)
theta2, cost_data = batch_gradient_decent(theta,X,y,200)
x = [i for i in range(len(cost_data))]
plt.plot(x,cost_data)
plt.title('Cost_J')
plt.show()