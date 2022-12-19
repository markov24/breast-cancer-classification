import numpy as np

# Create training data from raw numpy data
def generate_data():
    # Convert raw data into a NumPy array
    with open('data/tumor_data.txt', 'r') as f:
        data = np.array([[e for e in line.split(',')] for line in f])
    np.random.shuffle(data)

    X = np.array(data[:, 2:12], dtype=float)
    y = np.array(data[:, 1])
    for i in range(len(y)):
        if y[i] == 'M':
            y[i] = 1
        else:
            y[i] = 0
    y = np.array(y, dtype=float)
    return X, y