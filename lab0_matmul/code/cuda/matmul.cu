#include <cstdlib>
#include <cassert>
#include <iostream>
#include <string> // to_string

using namespace std;

int saved_h1 = -1, saved_w1 = -1, saved_w2 = -1;
float *g_mat1 = NULL, *g_mat2 = NULL, *g_matFinal = NULL;

// Below code to check it via nvcc in Linux
// It can be run as
// 		nvcc matmul.cu -o matmul
//		./matmul
// It should run without any issue


void init_matrix(float *mat, int h, int w) {
    for (int i = 0; i < h * w; i++) {
        mat[i] = rand() % 100 / 2.0f;
    }
}


void check_result(const float *mat1, const float *mat2, float *mat_final, int h1, int w1, int w2) {
    float tmp;
    for (int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j++) {
            tmp = 0;
            for (int k = 0; k < w1; k++) {
                tmp += mat1[i * h1 + k] * mat2[k * w1 + j];
            }
            assert(tmp == mat_final[i * h1 + j]);
        }
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


__global__ void matrixMul(const float *a, const float *b, float *c, int h1, int w1, int w2) {
	// Get row/column ID for the current thread
	//				 which block       which thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check bounds
	if (row < h1 && col < w2) {
		// Calculate result for single raw and col
		float accum = 0.0f;
		for (int i = 0; i < w1; i++) {
			accum += a[row * h1 + i] * b[i * w1 + col];
		}
		c[row * h1 + col] = accum;
	}
}


float gpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2, const int threads_x, const int threads_y) {
	if (h1 != saved_h1 || w1 != saved_w1 || w2 != saved_w2) {
		saved_h1 = h1;
		saved_w1 = w1;
		saved_w2 = w2;

		if (g_mat1 != NULL || g_mat2 != NULL || g_matFinal != NULL) {
			CHECK_ERROR(cudaFree(g_mat1));
			CHECK_ERROR(cudaFree(g_mat2));
			CHECK_ERROR(cudaFree(g_matFinal));
		}
		// Alloc
		CHECK_ERROR(cudaMalloc(&g_mat1,     h1 * w1 * sizeof(float)));
		CHECK_ERROR(cudaMalloc(&g_mat2,     w1 * w2 * sizeof(float)));
		CHECK_ERROR(cudaMalloc(&g_matFinal, h1 * w2 * sizeof(float)));
	}
    // Transfer the data to the device
    size_t size_mat1 =      h1 * w1 * sizeof(float);
    size_t size_mat2 =      w1 * w2 * sizeof(float);
    size_t size_mat_final = h1 * w2 * sizeof(float);
    CHECK_ERROR(cudaMemcpy(g_mat1, mat1, size_mat1, cudaMemcpyHostToDevice)); 
    CHECK_ERROR(cudaMemcpy(g_mat2, mat2, size_mat2, cudaMemcpyHostToDevice));

	// Kernel params
	dim3 THREADS(threads_x, threads_y);
	dim3 BLOCKS((h1 + THREADS.x - 1) / THREADS.x, (w2 + THREADS.y - 1) / THREADS.y);

	matrixMul<<<BLOCKS, THREADS>>>(g_mat1, g_mat2, g_matFinal, h1, w1, w2);
	CHECK_ERROR(cudaDeviceSynchronize());
    // Transfer the data to the host
    CHECK_ERROR(cudaMemcpy(finalMat, g_matFinal, size_mat_final, cudaMemcpyDeviceToHost));
}


float cpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2) {
	for (int i = 0; i < h1; i++) {
		for (int j = 0;j < w2; j++) {
			for (int k = 0; k < w1; k++) {
				finalMat[i * h1 + j] += mat1[i * h1 + k] * mat2[k * w1 + j];
			}
		}
	}
}


int main() {
    int h1 = 3, w1 = 3, h2 = 3, w2 = 3;
    float *mat1 = (float*)malloc(h1 * w1 * sizeof(float));
    float *mat2 = (float*)malloc(h2 * w2 * sizeof(float));
    float *mat_final = (float*)malloc(h1 * w2 * sizeof(float));

    init_matrix(mat1, h1, w1);
    init_matrix(mat2, h2, w2);
    gpu_matmul(mat1, mat2, mat_final, h1, w1, w2, 16, 32);
    check_result(mat1, mat2, mat_final, h1, w1, w2);
	fprintf(stderr, "Programm run with success\n");
    return 0; 
}