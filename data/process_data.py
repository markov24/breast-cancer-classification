import numpy as np

# Create training data from raw numpy data
def generate_data():
    # Convert raw data into a NumPy array
    with open('tumor_data.txt', 'r') as f:
        data = np.array([[e for e in line.split(',')] for line in f])

    # Schuffle data so train/test split is random
    np.random.shuffle(data)
    return data