float gpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2, const int threads, const int blocks);
float cpu_matmul(const float *mat1, const float *mat2, float *finalMat, const int h1, const int w1, const int w2);
