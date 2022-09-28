import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm

from timeit import timeit

from code import (matmul_numpy, matmul_one_thread_python, 
                    c_cuda_matmul, c_cpu_matmul, cpu_cython_matmul, cpu_cython_parallel_matmul)


def run_benchmark(array_size_start=3, array_size_end=12):
    """
        Launch a becnhamrk which measures average executing time
        for all implpemented matrix multiplication methods
    """
    array_size = [2 ** i for i in range(array_size_start, array_size_end)]
    results = {
        "matmul_numpy": [],
        "c_cuda_matmul": [],
        'matmul_one_thread_python': [],
        'cpu_cython_matmul': [],
        'c_cpu_matmul': [],
        'cpu_cython_parallel_matmul': [],
    }
    for N in tqdm(array_size[::]):
        mat1 = np.random.randn(N, N).astype(np.float32)
        mat2 = np.random.randn(N, N).astype(np.float32)
        mat_final = np.zeros((N, N), np.float32)

        if N < 512:
            results['matmul_one_thread_python'].append(
                timeit(
                    partial(matmul_one_thread_python, mat1, mat2, mat_final), 
                    number=5 # repeat
                ) / 5
            )
        else:
            # Otherwise it took huge amount of time
            results['matmul_one_thread_python'].append(0.0)

        if N <= 512:
            results['cpu_cython_matmul'].append(
                timeit(
                    partial(cpu_cython_matmul, mat1, mat2, mat_final),
                    number= 5 # repeat
                ) / 5
            )
            results['cpu_cython_parallel_matmul'].append(
                timeit(
                    partial(cpu_cython_parallel_matmul, mat1, mat2, mat_final, n_threads=6),
                    number=5 # repeat
                ) / 5
            )
            results['c_cpu_matmul'].append(
                timeit(
                    partial(c_cpu_matmul, mat1, mat2, mat_final),
                    number=5 # repeat
                ) / 5
            )
        else:
            # Otherwise it took huge amount of time
            results['cpu_cython_matmul'].append(0.0)
            results['c_cpu_matmul'].append(0.0)
            results['cpu_cython_parallel_matmul'].append(0.0)

        results['matmul_numpy'].append(
            timeit(
                partial(matmul_numpy, mat1, mat2, mat_final),
                number=100 # repeat
            ) / 100
        )
        results['c_cuda_matmul'].append(
            # Set different number of threads which depends of matrix shape
            # Otherwise cuda do not work well on small matrix
            timeit(
                partial(c_cuda_matmul, mat1, mat2, mat_final, min(N, 128), min(N, 128)),
                number=100 # repeat
            ) / 100
        )

    df = pd.DataFrame(results)
    df.to_csv("results.csv")


def perform_test(method, a, b, res, GT_result, **kwargs):
    """
        Checks method results equals to ground-truth result,
        otherwise raise AssertionError
    """
    method(a, b, res, **kwargs)
    assert np.allclose(GT_result, res), method.__name__ 
    res[:] = np.zeros(res.shape, res.dtype)


def check_matmul_correctness():
    """
        Check all methods with simple matrix
    """
    mat1 = np.random.randint(0, 10, (10, 10)).astype(np.float32)
    mat2 = np.random.randint(0, 10, (10, 10)).astype(np.float32)
    h1 = mat1.shape[0]
    w2 = mat2.shape[1]
    mat_final = np.zeros((h1, w2), dtype=np.float32)

    GT_result = np.dot(mat1, mat2)
    
    perform_test(matmul_numpy, mat1, mat2, mat_final, GT_result)
    perform_test(matmul_one_thread_python, mat1, mat2, mat_final, GT_result)

    perform_test(c_cuda_matmul, mat1, mat2, mat_final, GT_result, threads_x=4, threads_y=4)
    perform_test(cpu_cython_matmul, mat1, mat2, mat_final, GT_result)
    perform_test(c_cpu_matmul, mat1, mat2, mat_final, GT_result)
    perform_test(cpu_cython_parallel_matmul, mat1, mat2, mat_final, GT_result, n_threads=4)


if __name__ == "__main__":
    check_matmul_correctness()
    print("Check has been passed, all results are fine!")
    print("Run benchmark...")
    run_benchmark()

