from setuptools import Extension, setup
from Cython.Distutils import build_ext
import numpy as np

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext = Extension(
    'cythonRenderCpu',
    sources=['cythonRenderCpu.pyx'],
    libraries=['lib_cpu/cpurender'],  # , os.path.join(CUDA['lib64'], 'cudart')],
    language='c++',
    include_dirs=[numpy_include],
    library_dirs=['lib'],
    extra_compile_args=['-fopenmp', '-lgomp'],
    extra_link_args=['-fopenmp', '-lgomp']
)

setup(
    include_dirs=[numpy_include],
    ext_modules = [ext],
    cmdclass={'build_ext': build_ext},
)
