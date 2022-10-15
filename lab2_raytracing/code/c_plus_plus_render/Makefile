\PYTHON = python3
.PHONY = all help clean build run
.DEFAULT_GOAL = help

SRCS_CPU = code/raytracing_cpu.cc
OBJECTS_CPU = raytracing_cpu.o

help:
	@echo "----------HELP----------"
	@echo "HPC Lab2 Makefile."
	@echo "------------------------"

clean:
	@echo "----------CLEANING----------"
	rm -rf build && rm CythonRenderCpu.cpp CythonRenderCpu.cpython-38-x86_64-linux-gnu.so lib_cpu/*.a
	@echo "---------------------------"

build:
	@echo "----------BUILD CPU FILES----------"
	g++ -fopenmp -lgomp -fPIC -c  $(SRCS_CPU)
	ar rvs lib_cpu/libcpurender.a $(OBJECTS_CPU)
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
