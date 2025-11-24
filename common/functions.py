import numpy as np

def step_function0(x):
    return 1 if x > 0 else 0

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    print(step_function(x))
    print(sigmoid(x))
    print(relu(x))
    print(softmax(x))
    print(softmax(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])))
