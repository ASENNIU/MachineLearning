import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    data = sio.loadmat(path)
    return data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']

def cost(theta, X, y):
    """
    :param theta: R(n), linear regression parameters
    :param X: R(m*n), m records, n features
    """

    m = X.shape[0]
    theta = np.mat(theta).T
    inner = X @ theta - y
    cost = (inner.T @ inner) / (2 * m)

    return cost[0,0]

def gradient(theta, X, y):
    m = X.shape[0]
    theta = np.mat(theta).T
    _theta = X.T @ (X @ theta - y) / m

    return _theta

def regularized_gradient(theta, X, y, regRate=1):
    m = X.shape[0]
    _theta = theta.copy()

    _theta[0] = 0
    _theta = np.mat(_theta).T
    _theta = gradient(theta, X, y) + (regRate / m) * _theta

    return _theta

def regularized_cost(theta, X, y, regRate=1):
    m = X.shape[0]
    reg = (regRate/ (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + reg

def linear_regression(X, y ,regRate=1):
    theta = np.ones(X.shape[1])

    res = opt.minimize(fun=regularized_cost,x0=theta,args=(X,y,regRate),jac=regularized_gradient,method='TNC')

    return res

def normalize_feature(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

def prepare_poly_data(*args, power):

    return [prepare(X, power) for X in args]

def prepare(X, power):
    for i in range(2, power+1):
        X =np.insert(X, X.shape[1],np.power(X[:,0],i),axis=1)
    return normalize_feature(X)

def show_basicdata(X, y):
    plt.scatter(X,y,s=20)
    plt.xlabel('water_lavel')
    plt.ylabel('flow')
    plt.show()

def learning_curve(X, y, Xval, yval, regRate=1):
    training_cost, cv_cost = [], []
    m =X.shape[0]
    for i in range(1,m+1):
        res = linear_regression(X[:i,:], y[:i,:], regRate)
        tc = cost(res.x, X[:i,:], y[:i,:])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1,m+1), training_cost, label='training cost')
    plt.plot(np.arange(1,m+1), cv_cost, label='cv cost')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.legend()
    #plt.savefig('learning_curve.jpg')
    plt.show()

