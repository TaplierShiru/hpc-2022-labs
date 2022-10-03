import numpy as np


def vector_sum_one_thread_python(vector: np.ndarray):
    """
        Vector sum in Python using single thread,
        Can be very slow, if input matrix has more than 10^5 dim size
    """
    n = vector.shape[0]
    result = 0.0
    for i in range(n):
        result += vector[i]
    return result

def vector_sum_numpy(vector: np.ndarray):
    return np.sum(vector)
