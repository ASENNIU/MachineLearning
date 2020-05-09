import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta,X,y):
    first = np.multiply(y,np.log(0.00001+sigmoid(X @ theta)))
    second = np.multiply((1 - y), np.log(0.00001+1 - sigmoid(X @ theta)))
    return -np.sum(first + second) / X.shape[0]

def gradient_descent(theta,X,y):
    """
    grad = theta.copy()
    error = sigmoid(X * theta) - y
    for i in range(X.shape[1]):
        term = np.multiply(error, X[:, i])
        grad[i,0] = np.sum(term) / len(X)
    return grad
    """
    return X.T @ (sigmoid(X @ theta) - y) / X.shape[0]

def predict(theta,X):
    probability = sigmoid(X @ theta)
    return (probability >= 0.5).astype(int)

def costReg(theta,X,y,lr):
    first = np.multiply(y, np.log(0.00001 + sigmoid(X @ theta)))
    second = np.multiply((1 - y), np.log(0.00001 + 1 - sigmoid(X @ theta)))
    reg = (lr/(2*len(X))) * ((theta.T @ theta) -theta[0,0] * theta[0,0])
    return -np.sum(first + second) / len(X) + reg[0,0]

def gradientReg(theta,X,y,lr):
    _theta = theta.copy()
    _theta = X.T @ (sigmoid(X @ _theta) - y) + lr / len(X) * _theta
    _theta[0,0] -= lr / len(X) * theta[0,0]
    return _theta

