\PYTHON = python3
.PHONY = all help clean build run
.DEFAULT_GOAL = help

SRCS = code/raytracing_gpu.cu code/vec3.cu code/sphere.cu code/material.cu code/camera.cu
OBJECTS = raytracing_gpu.o vec3.o sphere.o material.o camera.o

SRCS_CPU = c_plus_plus_render/raytracing_cpu.cc
OBJECTS_CPU = raytracing_cpu.o

help:
	@echo "----------HELP----------"
	@echo "HPC Lab2 Makefile."
	@echo "------------------------"

clean:
	@echo "----------CLEANING----------"
	rm -rf build && rm CythonRenderGpu.cpp CythonRenderGpu.cpython-38-x86_64-linux-gnu.so lib_gpu/*.a
	@echo "---------------------------"

build:
	@echo "----------BUILD CUDA FILES----------"
	nvcc --compiler-options -fPIC --device-c $(SRCS) 
	nvcc --compiler-options -fPIC --device-link $(OBJECTS) -o link.o
	nvcc --lib $(OBJECTS) link.o -odir lib -o lib_gpu/libcudarender.a
	rm *.o
	@echo "---------------------------"
	
	@echo "----------BUILD CYTHON FILES----------"
	python3 setup.py build_ext --inplace
	@echo "---------------------------"

run:
	@echo "----------RUN----------"
	python3 -m main
	@echo "-------------------------"


all: build run
