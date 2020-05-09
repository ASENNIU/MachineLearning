import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import scipy.optimize as opt
from R_func import *

"""
Notes: X - num_movies (1682)  x num_features (10) matrix of movie features  
       Theta - num_users (943)  x num_features (10) matrix of user features  
       Y - num_movies x num_users matrix of user ratings of movies  
       R - num_movies x num_users matrix, where R(i, j) = 1 if the i-th movie was rated by the j-th user  
"""

movies_mat = sio.loadmat('data/ex8_movies.mat')
#print(movies_mat.keys())

Y, R = movies_mat.get('Y'), movies_mat.get('R')
# print(Y.shape, R.shape)

m, u = Y.shape
# m: num of movie
# u: num of users

n = 10
# how many feature for a movie

param_mat = sio.loadmat('data/ex8_movieParams.mat')
theta, X = param_mat.get('Theta'), param_mat.get('X')
# print(theta.shape, X.shape)

movie_list = []
with open('data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')

        movie_list.append(''.join(tokens[1:]))
movie_list = np.array(movie_list)

ratings = np.zeros(Y.shape[0])

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

Y = np.insert(Y, 0, ratings, axis=1)
R = np.insert(R, 0, ratings != 0, axis=1)

n_features = 10
n_movie, n_user = Y.shape
lr = 1

X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

param = serialize(X, theta)

Y_norm = Y - Y.mean(axis=1).reshape(Y.shape[0], 1)

res = opt.minimize(fun=regularized_cost, x0=param, args=(Y_norm, R, n_features, lr), method='TNC', jac=regularized_gradient)

X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)

prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean(axis=1)

idx = np.argsort(my_preds)[::-1]

for m in movie_list[idx][:10]:
    print(m)