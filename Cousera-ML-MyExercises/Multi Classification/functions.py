import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta,X,y,lr):
    theta = np.mat(theta).T
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(y,np.log(0.00001 + sigmoid(X @ theta)))
    secend = np.multiply((1-y),np.log(0.00001 + 1 - sigmoid(X @ theta)))
    reg = lr / (2 * len(X)) * (theta.T @ theta - theta[0,0] * theta[0,0])
    return -np.sum(first + secend) + reg

def gradient(theta,X,y,lr):
    theta = np.mat(theta).T
    X = np.mat(X)
    y = np.mat(y)
    _theta = theta.copy()
    _theta = X.T @ (sigmoid(X @ _theta) - y) / len(X) + (lr / len(X)) * _theta
    _theta[0,0] -= (lr / len(X)) * theta[0,0]
    return _theta

def one_vs_all(X,y,num_labels,lr):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))
    X = np.insert(X,0,values=np.ones(rows),axis=1)
    for i in range(1,num_labels+1):
        theta = np.random.randn(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(rows,1)
        #minimize the objective function
        fmin = minimize(fun=cost,x0=theta,args=(X,y_i,lr),method='TNC',jac=gradient)
        all_theta[i-1,:] = fmin.x
    return all_theta

def predict_all(X,all_theta):
    rows = X.shape[0]
    X =np.insert(X,0,values=np.ones(rows),axis=1)
    X = np.mat(X)
    all_theta = np.mat(all_theta)
    h = sigmoid(X @ all_theta.T)
    h_argmax = np.argmax(h,axis=1)
    return h_argmax + 1