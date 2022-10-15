#include "include/vec3.h"

__host__ __device__ vec3::vec3(float x, float y, float z)
{
    e[0] = x;
    e[1] = y;
    e[2] = z;
}

__host__ __device__ float vec3::x() const
{
    return e[0];
}

__host__ __device__ float vec3::y() const
{
    return e[1];
}

__host__ __device__ float vec3::z() const
{
    return e[2];
}

__host__ __device__ float vec3::r() const
{
    return e[0];
}

__host__ __device__ float vec3::g() const
{
    return e[1];
}

__host__ __device__ float vec3::b() const
{
    return e[2];
}

__host__ __device__ const vec3 &vec3::operator+() const
{
    return *this;
}

__host__ __device__ vec3 vec3::operator-() const
{
    return vec3(-e[0], -e[1], -e[2]);
}

__host__ __device__ float vec3::operator[](int i) const
{
    return e[i];
}

__host__ __device__ float &vec3::operator[](int i)
{
    return e[i];
}

__host__ __device__ vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator/=(const vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator-=(const vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator*=(const float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ vec3 &vec3::operator/=(const float t)
{
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ float vec3::length() const { 
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); 
}

__host__ __device__ float vec3::squared_length() const { 
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; 
}

__host__ __device__ void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__host__ __device__ vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ vec3 operator*(const vec3 &v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ float dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ vec3 unit_vector(vec3 v) {
    return v / v.length();
}
