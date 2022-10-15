#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float x, float y, float z);

    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;
    __host__ __device__ float r() const;
    __host__ __device__ float g() const;
    __host__ __device__ float b() const;
    
    __host__ __device__ const vec3& operator+() const;
    __host__ __device__ vec3 operator-() const;
    __host__ __device__ float operator[](int i) const;
    __host__ __device__ float& operator[](int i);

    __host__ __device__ vec3& operator+=(const vec3 &v2);
    __host__ __device__ vec3& operator-=(const vec3 &v2);
    __host__ __device__ vec3& operator*=(const vec3 &v2);
    __host__ __device__ vec3& operator/=(const vec3 &v2);
    __host__ __device__ vec3& operator*=(const float t);
    __host__ __device__ vec3& operator/=(const float t);

    __host__ __device__ float length() const;
    __host__ __device__ float squared_length() const;
    __host__ __device__ void make_unit_vector();

    float e[3];
};

__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 operator/(const vec3 &v1, const vec3 &v2);

__host__ __device__ vec3 operator*(float t, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &v, float t);
__host__ __device__ vec3 operator/(vec3 v, float t);

__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2);
__host__ __device__ float dot(const vec3 &v1, const vec3 &v2);
__host__ __device__ vec3 unit_vector(vec3 v);

#endif