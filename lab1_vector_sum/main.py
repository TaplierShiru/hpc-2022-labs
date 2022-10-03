import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm

from timeit import timeit

from code import (vector_sum_numpy, vector_sum_one_thread_python, 
                    c_cuda_vector_sum, c_cpu_vector_sum, cpu_cython_vector_sum)


def run_benchmark(array_size_start=8, array_size_end=29):
    """
        Launch a becnhamrk which measures average executing time
        for all implpemented matrix multiplication methods
    """
    array_size = [2 ** i for i in range(array_size_start, array_size_end)]
    results = {
        "vector_sum_numpy": [],
        "c_cuda_vector_sum": [],
        'vector_sum_one_thread_python': [],
        'cpu_cython_vector_sum': [],
        'c_cpu_vector_sum': [],
    }
    for N in tqdm(array_size[::]):
        vector = np.random.randn(N).astype(np.float32)

        if N <= 2 ** 21:
            results['vector_sum_one_thread_python'].append(
                timeit(
                    partial(vector_sum_one_thread_python, vector), 
                    number=5 # repeat
                ) / 5
            )
        else:
            # Otherwise it took huge amount of time
            results['vector_sum_one_thread_python'].append(0.0)

        results['cpu_cython_vector_sum'].append(
            timeit(
                partial(cpu_cython_vector_sum, vector),
                number= 5 # repeat
            ) / 5
        )
        results['c_cpu_vector_sum'].append(
            timeit(
                partial(c_cpu_vector_sum, vector),
                number=5 # repeat
            ) / 5
        )

        results['vector_sum_numpy'].append(
            timeit(
                partial(vector_sum_numpy, vector),
                number=100 # repeat
            ) / 100
        )
        results['c_cuda_vector_sum'].append(
            # Set different number of threads which depends of matrix shape
            # Otherwise cuda do not work well on small matrix
            timeit(
                partial(c_cuda_vector_sum, vector),
                number=100 # repeat
            ) / 100
        )

    df = pd.DataFrame(results)
    df.to_csv("results.csv")


def perform_test(method, vector, GT_result, **kwargs):
    """
        Checks method results equals to ground-truth result,
        otherwise raise AssertionError
    """
    res = method(vector, **kwargs)
    assert np.allclose(GT_result, res), method.__name__ 


def check_matmul_correctness():
    """
        Check all methods with simple matrix
    """
    vector = np.random.randint(0, 10, (100)).astype(np.float32)

    GT_result = np.sum(vector)
    
    perform_test(vector_sum_numpy, vector, GT_result)
    perform_test(vector_sum_one_thread_python, vector, GT_result)

    perform_test(c_cuda_vector_sum, vector, GT_result)
    perform_test(cpu_cython_vector_sum, vector, GT_result)
    perform_test(c_cpu_vector_sum, vector, GT_result)


if __name__ == "__main__":
    check_matmul_correctness()
    print("Check has been passed, all results are fine!")
    print("Run benchmark...")
    run_benchmark()

