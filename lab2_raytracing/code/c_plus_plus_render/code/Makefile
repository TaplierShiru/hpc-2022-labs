.PHONY = all help clean build run
.DEFAULT_GOAL = help

help:
	@echo "----------HELP----------"
	@echo "HPC Cuda Lab2 Makefile."
	@echo "------------------------"

clean:
	@echo "----------CLEANING----------"
	rm raytracing_cpu
	@echo "---------------------------"

build:
	@echo "----------BUILD CUDA FILES----------"
	g++ raytracing_cpu.cc -fopenmp -lgomp -o raytracing_cpu
	@echo "---------------------------"
	
run:
	@echo "----------RUN----------"
	./raytracing_cpu
	@echo "-------------------------"


all: build run

