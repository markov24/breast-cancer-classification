import numpy as np

# Build model with x^2 features
def square_model(X):
    m, n = X.shape
    X_squared = np.zeros((m, 2*n))
    X_squared[:, :n] = X
    for j in range(n):
        X_squared[:, n+j] = X[:, j]**2
    return X_squared

# Build model with x^2 and x^3 features
def cubic_model(X):
    m, n = X.shape
    X_cubic = np.zeros((m, 3*n))
    X_cubic[:, :n] = X
    for j in range(n):
        X_cubic[:, n+j] = X[:, j]**2
    for j in range(n):
        X_cubic[:, 2*n+j] = X[:, j]**3
    return X_cubic

# Normalize data using the mean and standard deviation
def normalize(X):
    mu = np.mean(X, axis=0)
    # mx = np.max(X, axis=0)
    # mn = np.min(X, axis=0)
    std = np.std(X, axis=0)
    return ((X - mu) / std), mu, std

# Generate training data with two optional parameters:
#   model: 1-standard, 2-squared, 3-cubic
#   train: train/test split, default of 90% training data
def training_data(X, y, features, train=0.9, model=1):
    # Select cutoff
    cutoff = round(train*X.shape[0])

    X_train = np.array(X[:cutoff, :features], dtype=float)
    X_test = np.array(X[cutoff:, :features], dtype=float)

    # Select chosen model and normalize
    if model==2:
        X_train = square_model(X_train)
    elif model==3:
        X_train = cubic_model(X_train)
    X_train = normalize(X_train)

    y_train = np.array(y[:cutoff])
    y_test = np.array(y[cutoff:])

    return X_train, X_test, y_train, y_test
    