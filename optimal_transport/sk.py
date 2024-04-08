import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

def sinkhorn(out, lam=10, sinkhorn_iterations=3):
    Q = np.exp(-lam * out).T 
    K = Q.shape[0] # how many prototypes
    B = Q.shape[1] # how many samples
    # make the matrix sums to 1
    sum_Q = np.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = np.sum(Q, axis=1, keepdims=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= np.sum(Q, axis=0, keepdims=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.T


X ,y = make_circles(n_samples=1000, noise=0.05, factor=0.5, shuffle=False)

X1 = X[y==0]
X2 = X[y==1]

n, m = len(X1), len(X2)

M = pairwise_distances(X1, X2, metric="euclidean")

P = sinkhorn(M)

fig, ax = plt.subplots()

ax.scatter(X1[:,0], X1[:,1], color='blue', label='set 1')
ax.scatter(X2[:,0], X2[:,1], color='orange', label='set 2')
ax.legend(loc=0)

plt.savefig("set.jpg")

r = np.ones(n) / n
c = np.ones(m) / m

alpha = 0.2
mixing = P.copy()
# Normalize, so each row sums to 1 (i.e. probability)
# mixing /= r.reshape((-1, 1))

X = (1 - alpha) * X1 + alpha * mixing @ X2
w = (1 - alpha) * r + alpha * mixing @ c

fig, ax = plt.subplots()

ax.scatter(X1[:,0], X1[:,1], color='blue', label='set 1')
ax.scatter(X2[:,0], X2[:,1], color='orange', label='set 2')

ax.scatter(X[:,0], X[:,1], color='red', label='interpolation')
ax.legend(loc=1)

plt.savefig('ie_sk.jpg')