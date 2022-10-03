cimport cython
from cython.parallel cimport prange, parallel


# Define function from static library
cdef extern from "cuda/vector_sum.h":
    float gpu_vector_sum(const float *vector, const int n)
    float cpu_vector_sum(const float *vector, const int n)


# Wrap C function
def c_cuda_vector_sum(float[:] vector):
    cdef:
        int n = vector.shape[0]
    return gpu_vector_sum(&vector[0], n)


def c_cpu_vector_sum(float[:] vector):
    cdef:
        int n = vector.shape[0]
    return cpu_vector_sum(&vector[0], n)


# Disable all checks to increase perfomance
@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
def cpu_cython_vector_sum(float[:] vector):
    cdef:
        int i
        float res = 0.0
    for i in range(vector.shape[0]):
        res += vector[i]
    return res

