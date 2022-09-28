import numpy as np


def matmul_one_thread_python(mat1: np.ndarray, mat2: np.ndarray, res: np.ndarray):
    """
        Matrix multiplication in Python using single thread,
        Can be very slow, if input matrix has more than 10^5 dim size
    """
    h1, w1 = mat1.shape
    h2, w2 = mat2.shape
    assert w1 == h2, f"Matrix can't be multiplied, Input shape mat1={mat1.shape} mat2={mat2.shape}"

    for h1_i in range(h1):
        for w2_i in range(w2):
            for w1_i in range(w1):
                res[h1_i, w2_i] += mat1[h1_i, w1_i] * mat2[w1_i, w2_i]


def matmul_numpy(a: np.ndarray, b: np.ndarray, res: np.ndarray):
    np.dot(a, b, res)
