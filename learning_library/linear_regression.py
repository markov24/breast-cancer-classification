import numpy as np

def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost = 0
    # Using least squares cost
    for i in range(m):
        cost += (w*x_train[i] + b - y_train[i])**2
    return (1/2*m)*cost

def compute_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]
    # dw, db represent partial derivatives of the Cost Function 
    # with respect to its parameters w and b
    dw, db = 0, 0
    for i in range(m):
        dw += (w*x_train[i] + b - y_train[i])*x_train[i]
        db += (w*x_train[i] + b - y_train[i])
    return dw/m, db/m

def gradient_descent(iter, x_train, y_train, a):
    w, b, dw, db = 0, 0, 0, 0
    for i in range(iter):
        dw, db = compute_gradient(x_train, y_train, w, b)
        w = w - a*dw
        b = b - a*db
    return w, b
