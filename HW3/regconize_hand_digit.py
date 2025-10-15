import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

x_all, y_all = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

x0 = x_all[np.where(y_all == '0')[0]]
x1 = x_all[np.where(y_all == '1')[0]]

plt.imshow(x1[100].reshape(28, 28))
plt.imshow(x0[100].reshape(28, 28))

x0 = x0[:1000, :]
x1 = x1[:1000, :]

y0 = np.zeros((x0.shape[0]))
y1 = np.ones((x1.shape[0]))

X = np.concatenate((x0, x1), axis = 0)
Y = np.concatenate((y0, y1), axis = 0)

one = np.ones((X.shape[0], 1))
X = np.concatenate((X, one), axis = 1)

def sigmoid(s):
    s = np.clip(s, -500, 500)
    return 1 / (1 + np.exp(-s))

def sgd(X, Y, theta_ini, eta = 0.05, gamma = 0.9):
    theta_old = theta_ini
    theta_epoch = theta_old
    v_old = np.zeros_like(theta_ini)
    N = X.shape[0]

    for it in range(1000):
        mix_id = np.random.permutation(N) 

        for i in mix_id:
            xi = X[i, :]
            yi = Y[i]
            hi = sigmoid(np.dot(xi, theta_old.T))
            gi = (hi - yi) * xi

            v_new = gamma * v_old + eta * gi

            theta_new = theta_old - v_new

            theta_old = theta_new
            v_old = v_new

        if(np.linalg.norm(theta_epoch - theta_old) < 1e-3):
            break

        theta_epoch = theta_old

    return (theta_epoch, it)

theta_init = np.random.rand(1, X.shape[1])[0]

(theta, it) = sgd(X, Y, theta_init)
print("Theta=", theta, "Interation=", it)

np.savetxt('theta.txt', theta)