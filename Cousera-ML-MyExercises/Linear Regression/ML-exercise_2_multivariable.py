import pandas as pd
import matplotlib.pyplot as plt
from functions import *

data = pd.read_csv('data/houses.txt',header=None,sep='\t',names=['Size','Bedrooms','Price'])
data = (data - data.mean()) / data.std()
print(data.head(5))
data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.random.randn(X.shape[1])).T


theta_t, cost_list = gradientDescent(X,y,theta)
theta_Eqn = normalEqn(X,y)
x = [i for i in range(len(cost_list))]
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(x,cost_list)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
print(theta_t)
print(theta_Eqn)

