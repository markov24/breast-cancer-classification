import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function: domain: R -> range: (0, 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Least squares cost is not convex, hence we use a piecewise cost function
def compute_cost(X_train, y_train, w_vec, b, l):
    m, n = X_train.shape
    cost = 0
    for i in range(m):
        z = sigmoid(np.dot(X_train[i], w_vec) + b)
        cost += y_train[i]*np.log(z) + (1-y_train[i])*np.log(1-z)
    cost *= (-1/m)
    cost_reg = 0
    for j in range(n):
        cost_reg += w_vec[j]**2
    cost_reg *= (l/(2*m))
    return cost + cost_reg

def compute_gradient(X_train, y_train, w_vec, b, l):
    m, n = X_train.shape
    # dw_vec, db represent derivatives of the Cost Function 
    # with respect to its parameters w (vector) and b
    dw_vec = np.zeros(n)
    db = 0
    for i in range(m):
        dhelp = sigmoid(np.dot(X_train[i], w_vec) + b) - y_train[i]
        db += dhelp
        for j in range(n):
            dw_vec[j] += dhelp*X_train[i, j]
    for j in range(n):
        dw_vec[j] += (l/m)*w_vec[j]
    return (1/m)*dw_vec, db/m

def gradient_descent(iter, X_train, y_train, a, l):
    m, n = X_train.shape
    cost_vec = []
    iter_vec = []
    b, db = 0, 0
    w_vec = np.zeros(n)
    dw_vec = np.zeros(n)
    # Print out status 10 times
    for i in range(iter):
        iter_vec.append(i)
        cost = compute_cost(X_train, y_train, w_vec, b, l)
        cost_vec.append(cost)
        dw_vec, db = compute_gradient(X_train, y_train, w_vec, b, l)
        w_vec = w_vec - a*dw_vec
        b = b - a*db
        if i%(iter//10) == 0:
            print(f"Iteration: {i}  Cost: {cost}")
    iter_vec, cost_vec = iter_vec[1:], cost_vec[1:]
    plt.plot(iter_vec, cost_vec)
    plt.show()
    return w_vec, b

def predict(X_test, y_test, w_vec, b):
    print(f"Parameters of model are: {w_vec} and {b}")
    m, n = X_test.shape
    correct = 0
    for i in range(m):
        z = np.dot(X_test[i], w_vec) + b
        prediction = sigmoid(z)
        if prediction >= 0.5 and y_test[i] == 1:
            correct += 1
        elif prediction < 0.5 and y_test[i] == 0:
            correct += 1
    print(f"Model accuracy is {round(100*correct/m, 3)}%")
