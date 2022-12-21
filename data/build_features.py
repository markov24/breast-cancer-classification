import numpy as np

# Build model containing all possible combinations of squared features
def square_model(X):
    m, n = X.shape
    rows = int(n + (n*(n+1))/2)
    X_squared = np.zeros((m, rows))
    X_squared[:, :n] = X
    counter = n
    for i in range(n):
        for j in range(i, n):
            X_squared[:, counter] = np.multiply(X[:, i], X[:, j])
            counter += 1    
    return X_squared

# Build model containing all possible combinations of squared and cubed features
def cubic_model(X):
    m, n = X.shape
    rows_2 = int(n+(n*(n+1))/2)
    rows_3 = int(rows_2 + (n*(n+1)*(n+2))/6)
    X_cubed = np.zeros((m, rows_3))
    X_cubed[:, :rows_2] = square_model(X)
    counter = rows_2
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                X_cubed[:, counter] = np.multiply(X[:, i], X[:, j], X[:, k])
                counter += 1
    return X_cubed
    

# Normalize data using the mean and standard deviation
def normalize(X, mu, std):
    return (X - mu) / std

# Generate training data with two optional parameters:
#   model: 1-standard, 2-squared, 3-cubic
#   train: train/test split, default of 90% training data
def training_data(X, y, features, train=0.9, model=1):
    # Select features
    X = X[:, features]

    # Select chosen model
    if model==2:
        X = square_model(X)
    elif model==3:
        X = cubic_model(X)

    # Calculate train/test split cutoff value
    cutoff = round(train*X.shape[0])
    X_train = np.array(X[:cutoff, :], dtype=float)
    X_test = np.array(X[cutoff:, :], dtype=float)

    # Calculate mean and std of X_train
    mu = np.mean(X, axis=0)
    # mx = np.max(X, axis=0)
    # mn = np.min(X, axis=0)
    std = np.std(X, axis=0)

    # Normalize X_train and X_test using mean and std of X_train
    X_train = normalize(X_train, mu, std)
    X_test = normalize(X_test, mu, std)

    # Obtain y_train and y_test using cutoff value
    y_train = np.array(y[:cutoff])
    y_test = np.array(y[cutoff:])

    return X_train, X_test, y_train, y_test