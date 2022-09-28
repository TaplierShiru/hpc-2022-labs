cimport cython
from cython.parallel cimport prange, parallel


# Define function from static library
cdef extern from "cuda/matmul.h":
    float gpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2, const int threads_x, const int threads_y)
    float cpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2)


# Wrap C function
def c_cuda_matmul(float[:, :] mat1, float[:, :] mat2, float[:, :] mat_final, int threads_x = 128, int threads_y = 128):
    cdef:
        int h1 = mat1.shape[0]
        int w1 = mat1.shape[1]
        int w2 = mat2.shape[1]
    return gpu_matmul(&mat1[0, 0], &mat2[0, 0], &mat_final[0, 0], h1, w1, w2, threads_x, threads_y)


def c_cpu_matmul(float[:, :] mat1, float[:, :] mat2, float[:, :] mat_final):
    cdef:
        int h1 = mat1.shape[0]
        int w1 = mat1.shape[1]
        int w2 = mat2.shape[1]
    return cpu_matmul(&mat1[0, 0], &mat2[0, 0], &mat_final[0, 0], h1, w1, w2)


# Disable all checks to increase perfomance
@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
def cpu_cython_matmul(float[:, :] mat1, float[:, :] mat2, float[:, :] mat_final):
    cdef:
        int i, j, k
        float res
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[1]):
            res = 0.0
            for k in range(mat1.shape[1]):
                res = res + mat1[i, k] * mat2[k, j]
            mat_final[i, j] = res


@cython.boundscheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
def cpu_cython_parallel_matmul(float[:, :] mat1, float[:, :] mat2, float[:, :] mat_final, int n_threads):
    cdef:
        int i, j, k
        float res
    # parallel and prange - OpenMP functions
    with nogil, parallel(num_threads=n_threads):
        for i in prange(mat1.shape[0], schedule='static'):
            for j in range(mat2.shape[1]):
                res = 0.0
                for k in range(mat1.shape[1]):
                    res = res + mat1[i, k] * mat2[k, j]
                mat_final[i, j] = res