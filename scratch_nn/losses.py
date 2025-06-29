import numpy as np

def cross_entropy(output_a, y):
    return -np.sum( y * np.log(output_a + 1e-12)) / y.shape[0]

def cross_entropy_delta(probs, labels):
    return (probs - labels) / labels.shape[0]