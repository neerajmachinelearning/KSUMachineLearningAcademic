import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('home.csv', names=["size","bedroom","price"])
# print(data)


data = (data - data.mean())/data.std()

# print(data)

X = data.iloc[:, 0:2]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

y = data.iloc[:, 2]


alpha = 0.01 #learning rate
iters = 1000 #iterations

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


# gradient descent
def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    return theta, cost

theta = np.zeros([1,3])
print(f"theta value: {theta}")
# running the gd and cost function
g, cost = gradientDescent(X, y, theta, iters, alpha)
print(g)

finalCost = computeCost(X, y, g)
print(finalCost)


#plot the cost
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')