import numpy as np

def computeCost(X, y, theta):
    inner = np.power((X @ theta - y),2)
    return np.sum(inner)/(2*X.shape[0])

def gradient( X, y,theta):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m

def gradientDescent(X, y, theta, alpha=0.01, iters=5000):
    _theta = theta.copy()
    cost = []
    for i in range(iters):
        _theta = _theta - alpha * gradient(X,y,_theta)
        cost.append(computeCost(X,y,_theta))
    return _theta, cost

def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta