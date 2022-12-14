#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include <curand_kernel.h>
#include "ray.h"
#include "hittable.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))


__device__ float schlick(float cosine, float ref_idx);

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state);

__device__ vec3 reflect(const vec3& v, const vec3& n);


class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
        __device__ virtual vec3 emitted(double u, double v, vec3 &p) const { return vec3(0.0, 0.0, 0.0); }
};


class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const;

        vec3 albedo;
};


class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const;

        vec3 albedo;
        float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const;

    float ref_idx;
};

class diffuse_light : public material  {
    public:
        __device__ diffuse_light(vec3 c) : color(c) {}

        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state
        ) const override {
            return false;
        }

        __device__ virtual vec3 emitted(double u, double v, vec3& p) const override {
            return color;
        }

        vec3 color;
};

#endif
