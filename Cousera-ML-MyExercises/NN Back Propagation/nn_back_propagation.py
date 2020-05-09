import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from functions import *
from scipy.optimize import minimize

data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']


#one-hot编码
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})

X = np.mat(X)
theta1 = np.mat(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.mat(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(correct) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))