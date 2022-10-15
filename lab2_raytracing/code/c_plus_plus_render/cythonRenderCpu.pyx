cimport cython
from libcpp cimport bool

cimport numpy as cnp


cdef extern from "code/raytracing_cpu.h":
    int render_scene(unsigned char* img, const int image_height, const int image_width, const bool isBigScene);
    int render_scene_parallel(unsigned char* img, const int image_height, const int image_width, const bool isBigScene);


def cpu_render_scene(cnp.ndarray[unsigned char, ndim=3] image, int image_width, int image_height, bool isBigScene):
    return render_scene(&image[0, 0, 0], image_height, image_width, isBigScene)


def cpu_render_scene_parallel(cnp.ndarray[unsigned char, ndim=3] image, int image_width, int image_height, bool isBigScene):
    return render_scene_parallel(&image[0, 0, 0], image_height, image_width, isBigScene)
