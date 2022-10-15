.PHONY = all help clean build run
.DEFAULT_GOAL = help

SRCS = raytracing_gpu.cu vec3.cu sphere.cu material.cu camera.cu
OBJS = raytracing_gpu.o vec3.o sphere.o material.o camera.o

help:
	@echo "----------HELP----------"
	@echo "HPC Cuda Lab2 Makefile."
	@echo "------------------------"

clean:
	@echo "----------CLEANING----------"
	rm raytracing_gpu
	@echo "---------------------------"

build:
	@echo "----------BUILD CUDA FILES----------"
	nvcc --compiler-options -fPIC --device-c $(SRCS) 
	nvcc $(OBJS) -o raytracing_gpu
	rm *.o
	@echo "---------------------------"
	
run:
	@echo "----------RUN----------"
	./raytracing_gpu
	@echo "-------------------------"


all: build run

