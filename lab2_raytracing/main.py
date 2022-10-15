import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import skimage.io as io

from timeit import timeit

from code import cuda_render, clear_gpu_memory, cpu_render_scene, cpu_render_scene_parallel


def run_benchmark(array_size_start=3, array_size_end=8):
    """
        Launch a becnhamrk which measures average executing time
        for all implpemented raytracing methods
    """
    array_size = [2 ** i for i in range(array_size_start, array_size_end)]
    results = {
        "cuda_render": [],
        "cuda_render_big_scene": [],
        
        "cpu_render_scene": [],
        "cpu_render_scene_big_scene": [],

        'cpu_render_scene_parallel': [],
        'cpu_render_scene_parallel_big_scene': [],
    }
    for N in tqdm(array_size[::]):
        image = np.zeros((N, N, 3), dtype=np.uint8)
        image = np.ascontiguousarray(image)

        results['cuda_render_big_scene'].append(
            timeit(
                partial(cuda_render, image, N, N, True),
                number=3 # repeat
            ) / 3
        )
        results['cuda_render'].append(
            timeit(
                partial(cuda_render, image, N, N, False),
                number=5 # repeat
            ) / 5
        )
        
        results['cpu_render_scene_big_scene'].append(
            timeit(
                partial(cpu_render_scene, image, N, N, True),
                number=3 # repeat
            ) / 3
        )        
        results['cpu_render_scene'].append(
            timeit(
                partial(cpu_render_scene, image, N, N, False),
                number=5 # repeat
            ) / 5
        )

        results['cpu_render_scene_parallel_big_scene'].append(
            # Set different number of threads which depends of matrix shape
            # Otherwise cuda do not work well on small matrix
            timeit(
                partial(cpu_render_scene_parallel, image, N, N, True),
                number=3 # repeat
            ) / 3
        )
        results['cpu_render_scene_parallel'].append(
            # Set different number of threads which depends of matrix shape
            # Otherwise cuda do not work well on small matrix
            timeit(
                partial(cpu_render_scene_parallel, image, N, N, False),
                number=5 # repeat
            ) / 5
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


def check_raytracing():
    """
        Check all methods with simple matrix
    """
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image = np.ascontiguousarray(image)

    cuda_render(image, 128, 128, False)
    io.imsave('images/cuda_test.png', image[::-1])
    cuda_render(image, 128, 128, True)
    io.imsave('images/cuda_test_big_scene.png', image[::-1])

    clear_gpu_memory()

    cpu_render_scene(image, 128, 128, False)
    io.imsave('images/cpu_test.png', image[::-1])
    cpu_render_scene(image, 128, 128, True)
    io.imsave('images/cpu_test_big_scene.png', image[::-1])

    cpu_render_scene_parallel(image, 128, 128, False)
    io.imsave('images/cpu_parallel_test.png', image[::-1])
    cpu_render_scene_parallel(image, 128, 128, True)
    io.imsave('images/cpu_parallel_test_big_scene.png', image[::-1])



if __name__ == "__main__":
    check_raytracing()
    print("You can check images, in lab folder for each method!")
    print("Run benchmark...")
    run_benchmark()

