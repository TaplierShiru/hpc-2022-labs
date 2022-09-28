\PYTHON = python3
.PHONY = all clean build run
.DEFAULT_GOAL = all

clean:
	@echo "Clean..."
	rm cythonCpuMatMulWrapper.cpp cythonCpuMatMulWrapper.cpython-38-x86_64-linux-gnu.so && rm -rf build && rm -rf lib
	@echo "Clean done!"

build:
	@echo "Build Cython files..."
	mkdir lib
	nvcc --compiler-options -fPIC -lib -odir lib -o lib/libmatmul.a cuda/matmul.cu 
	python3 setup.py build_ext --inplace
	@echo "Building of Cython files done!"

run:
	@echo "Run..."
	python3 Main.py
	@echo "Run is done!"


all: build run
