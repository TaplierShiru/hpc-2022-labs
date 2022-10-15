#include "include/camera.h"


__device__ ray camera::get_ray(float s, float t, curandState *local_rand_state) {
    vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
    vec3 offset = u * rd.x() + v * rd.y();
    return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
}

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}