#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>    // std::max
#include <string> // to_string
#include <curand_kernel.h>
#include <float.h>


#include "include/vec3.h"
#include "include/ray.h"
#include "include/hittable.h"
#include "include/sphere.h"
#include "include/hittable_list.h"
#include "include/material.h"
#include "include/camera.h"

#include "include/color.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "include/stb_image_write.h"

using namespace std;

#define NUM_THREADS 1024;

int saved_w = -1, saved_h = -1, saved_num_hittable = -1;
bool saved_isBigScene = false;

vec3 *g_result_fb = NULL, *g_fb = NULL;
hittable **g_d_list = NULL, **g_d_world = NULL;
camera **g_d_camera = NULL;

curandState *g_d_rand_state;
curandState *g_d_rand_state2;


#define CHECK_ERROR(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hittable **world, int depth, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < depth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, int depth, camera **cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2f, b + RND);
                if(choose_mat < 0.8f) {
                    // diffuse
                    lambertian *albedo = new lambertian(vec3(RND*RND, RND*RND, RND*RND));
                    d_list[i++] = new sphere(center, 0.2, albedo);
                }
                else if(choose_mat < 0.95f) {
                    // metal
                    vec3 albedo = vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND));
                    float fuzz = 0.5f*RND;
                    metal *material = new metal(albedo, fuzz);
                    d_list[i++] = new sphere(center, 0.2, material);
                }
                else {
                    // glass
                    dielectric *material = new dielectric(1.5);
                    d_list[i++] = new sphere(center, 0.2, material);
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0, // fov
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}


__global__ void create_world_small(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        d_list[1] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[2] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[3] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *d_world  = new hittable_list(d_list, 4);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hittable **d_list, int num_hittables) {
    for(int i=0; i < num_hittables; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
}

void clear_cuda() {
    // clean up
    free_world<<<1,1>>>(g_d_list, saved_num_hittable);
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaFree(g_d_camera));
    CHECK_ERROR(cudaFree(g_d_world));
    CHECK_ERROR(cudaFree(g_d_list));
    CHECK_ERROR(cudaFree(g_d_rand_state));
    CHECK_ERROR(cudaFree(g_d_rand_state2));
    CHECK_ERROR(cudaFree(g_fb));
}


int gpu_render(unsigned char *img_fb, const int image_height, const int image_width, const bool isBigScene) {
    int tx = 16;
    int ty = 16;
    int ns = 20;
    int depth = 50;
	if (image_height != saved_h || image_width != saved_w || saved_isBigScene != isBigScene) {
		saved_h = image_height;
        saved_w = image_width;
        saved_isBigScene = isBigScene;

        int num_hittables;
        if (isBigScene) {
            num_hittables = 22*22+1+3;
        } else {
            num_hittables = 4;
        }
        saved_num_hittable = num_hittables;

        if (g_result_fb != NULL || g_fb != NULL || g_d_list != NULL || g_d_world != NULL) {
            // clear_cuda();
        }
		// Alloc
        g_result_fb = (vec3*)malloc(image_width*image_height * sizeof(vec3));
        CHECK_ERROR(cudaMalloc(&g_fb, image_width*image_height*sizeof(vec3)));
        // allocate random state
        CHECK_ERROR(cudaMalloc((void **)&g_d_rand_state, image_width*image_height*sizeof(curandState)));
        CHECK_ERROR(cudaMalloc((void **)&g_d_rand_state2, 1*sizeof(curandState)));

        // we need that 2nd random state to be initialized for the world creation
        rand_init<<<1,1>>>(g_d_rand_state2);
        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());

        // make our world of hittables & the camera
        if (isBigScene) {
            CHECK_ERROR(cudaMalloc((void **)&g_d_list, num_hittables*sizeof(hittable *)));
            CHECK_ERROR(cudaMalloc((void **)&g_d_world, sizeof(hittable *)));
            CHECK_ERROR(cudaMalloc((void **)&g_d_camera, sizeof(camera *)));
            create_world<<<1,1>>>(g_d_list, g_d_world, g_d_camera, image_width, image_height, g_d_rand_state2);
        } else {
            CHECK_ERROR(cudaMalloc((void **)&g_d_list, num_hittables*sizeof(hittable *)));
            CHECK_ERROR(cudaMalloc((void **)&g_d_world, sizeof(hittable *)));
            CHECK_ERROR(cudaMalloc((void **)&g_d_camera, sizeof(camera *)));
            create_world_small<<<1,1>>>(g_d_list, g_d_world, g_d_camera, image_width, image_height);
        }
        CHECK_ERROR(cudaGetLastError());
        CHECK_ERROR(cudaDeviceSynchronize());
	}
    // Render our buffer
    dim3 blocks(image_width/tx+1, image_height/ty+1);
    dim3 threads(tx, ty);
	render_init<<<blocks, threads>>>(image_width, image_height, g_d_rand_state);
	CHECK_ERROR(cudaGetLastError());
	CHECK_ERROR(cudaDeviceSynchronize());
	render<<<blocks, threads>>>(g_fb, image_width, image_height, ns, depth, g_d_camera, g_d_world, g_d_rand_state);
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
    // Transfer the data to the host
    CHECK_ERROR(cudaMemcpy(g_result_fb, g_fb, image_height*image_width * sizeof(vec3), cudaMemcpyDeviceToHost));

	for (int i = 0, k = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++, k++) {
            write_color(img_fb, g_result_fb[k], ns, i, j, image_height, image_width, 3);
        }
	}
}


int main() {
	const int image_width = 256;
	const int image_height = 256;

    unsigned char* img_fb = (unsigned char*)malloc(image_height * image_width * 3 * sizeof(unsigned char));
    gpu_render(img_fb, image_height, image_width, true);
	char const *filename_bin_scene = "image_bin_scene.bmp";
	stbi_write_bmp(filename_bin_scene, image_width, image_height, 3, img_fb);
    
    gpu_render(img_fb, image_height, image_width, false);
	char const *filename = "image.bmp";
	stbi_write_bmp(filename, image_width, image_height, 3, img_fb);
    
    clear_cuda();
    cudaDeviceReset();
	fprintf(stderr, "Programm run with success\n");
    return 0; 
}