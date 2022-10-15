cimport cython
from libcpp cimport bool

cimport numpy as cnp



cdef extern from "code/raytracing_gpu.h":
    int gpu_render(unsigned char* img, const int image_height, const int image_width, const bool isBigScene);
    void clear_cuda()

    
def cuda_render(cnp.ndarray[unsigned char, ndim=3] image, int image_width, int image_height, bool isBigScene):
    return gpu_render(&image[0, 0, 0], image_height, image_width, isBigScene)


def clear_gpu_memory():
    clear_cuda()

