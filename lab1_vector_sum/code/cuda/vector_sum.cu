#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>    // std::max
#include <string> // to_string

using namespace std;

#define NUM_THREADS 1024;

int saved_n = -1;
float *g_vector = NULL, *g_result_vector = NULL, *g_temp_result_vector = NULL;


// Good course about reduce sum using CUDA
// https://www.youtube.com/watch?v=Qpx227w6idA&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=13
//

// Below code to check it via nvcc in Linux
// It can be run as
// 		nvcc vector_sum.cu -o vector_sum
//		./vector_sum
// It should run without any issue


void init_matrix(float *vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = 1; //rand() % 100 / 2.0f;
    }
}

// End utils
//

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// For last iteration (saves useless work)
// Use volatile to prevent caching in registers (compiler optimization)
// No __syncthreads() necessary!
__device__ void warpReduce(volatile float* shmem_ptr, int t) {
	shmem_ptr[t] += shmem_ptr[t + 32];
	shmem_ptr[t] += shmem_ptr[t + 16];
	shmem_ptr[t] += shmem_ptr[t + 8];
	shmem_ptr[t] += shmem_ptr[t + 4];
	shmem_ptr[t] += shmem_ptr[t + 2];
	shmem_ptr[t] += shmem_ptr[t + 1];
}

__global__ void sum_reduction(float *v, float *v_r) {
	// Allocate shared memory
	// Its better to put `NUM_THREADS` defined var here (instead of 1024), 
	// but its does not compile with it. Do not know why...
	__shared__ float partial_sum[1024];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	// Stop early (call device function instead)
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(partial_sum, threadIdx.x);
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}


float gpu_vector_sum(const float *vector, const int n) {
	if (n != saved_n) {
		saved_n = n;

		if (g_vector != NULL || g_result_vector != NULL) {
			CHECK_ERROR(cudaFree(g_vector));
			CHECK_ERROR(cudaFree(g_result_vector));
		}
		// Alloc
		CHECK_ERROR(cudaMalloc(&g_vector, n * sizeof(float)));
		CHECK_ERROR(cudaMalloc(&g_result_vector, n * sizeof(float)));
		g_temp_result_vector = (float*)malloc(n * sizeof(float));
	}
    // Transfer the data to the device
    size_t size_vector = n * sizeof(float);
    CHECK_ERROR(cudaMemcpy(g_vector, vector, size_vector, cudaMemcpyHostToDevice));

	// Kernel params
	int THREADS = NUM_THREADS;
	int BLOCKS = max(n / THREADS / 2, 1); // If n << 1024, number of block can be zero, so assign atleast one block, if array too small

	sum_reduction<<<BLOCKS, THREADS>>>(g_vector, g_result_vector);
	sum_reduction<<<1,      THREADS>>>(g_result_vector, g_result_vector);
	CHECK_ERROR(cudaDeviceSynchronize());
    // Transfer the data to the host
    CHECK_ERROR(cudaMemcpy(g_temp_result_vector, g_result_vector, size_vector, cudaMemcpyDeviceToHost));
	return g_temp_result_vector[0];
}


float cpu_vector_sum(const float *vector, const int n) {
	float result = 0.0f;
	for (int i = 0; i < n; i++) {
		result += vector[i];
	}
	return result;
}


int main() {
    int n = 1 << 16;
    float *vector = (float*)malloc(n * sizeof(float));
    init_matrix(vector, n);

	float cpu_result = cpu_vector_sum(vector, n);
	float gpu_result = gpu_vector_sum(vector, n);
	fprintf(stderr, "Cpu result: %f\n", cpu_result);
	fprintf(stderr, "Gpu result: %f\n", gpu_result);
	assert(cpu_result == gpu_result);
	fprintf(stderr, "Programm run with success\n");
    return 0; 
}