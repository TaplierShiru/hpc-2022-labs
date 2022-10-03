\PYTHON = python3
.PHONY = all clean build
.DEFAULT_GOAL = all

clean:
	@echo "Clean..."
	rm cythonCpuVectorSumWrapper.cpp cythonCpuVectorSumWrapper.cpython-38-x86_64-linux-gnu.so && rm -rf build && rm -rf lib
	@echo "Clean done!"

build:
	@echo "Build Cython files..."
	mkdir lib
	nvcc --compiler-options -fPIC -lib -odir lib -o lib/libvector_sum.a cuda/vector_sum.cu 
	python3 setup.py build_ext --inplace
	@echo "Building of Cython files done!"

all: build
