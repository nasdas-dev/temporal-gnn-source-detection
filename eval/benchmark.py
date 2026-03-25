import numpy as np

def average_rank(possible):
    average_rank = (np.sum(possible, axis=1) + 1) / 2
    return average_rank

def sampled_rank(possible):
    row_sums = np.sum(possible, axis=1)
    return np.array([np.random.randint(1, s + 1) for s in row_sums])

def uniform_probabilities(possible):
    return possible / np.sum(possible, axis=1, keepdims=True)

