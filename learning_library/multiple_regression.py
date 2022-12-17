import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X_train, y_train, w_vec, b):
    m = X_train.shape[0]
    cost = 0
    for i in range(m):
        cost += (np.dot(w_vec, X_train[i]) + b - y_train[i])**2
    return (1/2*m)*cost

def compute_gradient(X_train, y_train, w_vec, b):
    m, n = X_train.shape
    dw_vec = np.zeros(n)
    db = 0
    for i in range(m):
        dhelp = np.dot(w_vec, X_train[i]) + b - y_train[i]
        db += dhelp
        for j in range(n):
            dw_vec[j] += dhelp*X_train[i, j]
    return (1/m)*dw_vec, db/m

def gradient_descent(iter, X_train, y_train, a):
    cost_vec = []
    iter_vec = []
    b, db = 0, 0
    w_vec = np.zeros(X_train.shape[1])
    dw_vec = np.zeros(X_train.shape[1])
    for i in range(iter):
        if i%(iter//100) == 0:
            iter_vec.append(i)
            cost_vec.append(compute_cost(X_train, y_train, w_vec, b))
            print(f"Iteration: {i}")
        dw_vec, db = compute_gradient(X_train, y_train, w_vec, b)
        w_vec = w_vec - a*dw_vec
        b = b - a*db
    iter_vec, cost_vec = iter_vec[1:], cost_vec[1:]
    plt.plot(iter_vec, cost_vec)
    plt.show()
    return w_vec, b
    