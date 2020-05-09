import numpy as np

def serialize(X, theta):
    """
    X(movie, feature)
    theta(user, feature)
    """

    return np.concatenate((X.ravel(), theta.ravel()))

def deserialize(param, n_movie, n_user, n_features):
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
        param[n_movie * n_features:].reshape(n_user, n_features)

def cost(param, Y, R, n_features):
    # compute cost for every r(i, j)=1
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2

def gradient(param, Y, R, n_features):
    # theta (user, feature)
    # X (movie, feature)
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    X_grad = inner @ theta
    theta_grad = inner.T @ X

    return serialize(X_grad, theta_grad)

def regularized_cost(param, Y, R, n_feartures, lr=1):
    reg = np.power(param, 2).sum() * (lr / 2)
    return cost(param, Y, R, n_feartures) + reg

def regularized_gradient(param, Y, R, n_features, lr=1):
    grad = gradient(param, Y, R, n_features)
    return grad + lr * param