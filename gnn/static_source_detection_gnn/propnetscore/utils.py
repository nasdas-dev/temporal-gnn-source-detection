import numpy as np

def standard_basis_vec(size, i):
    e = np.zeros(size)
    e[i] = 1
    return e
