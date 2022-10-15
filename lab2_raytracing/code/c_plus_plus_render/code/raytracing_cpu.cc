#include "omp.h"

#include "render_utils/rtweekend.h"
#include "render_utils/color.h"
#include "render_utils/hittable_list.h"
#include "render_utils/sphere.h"
#include "render_utils/camera.h"
#include "render_utils/material.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "render_utils/stb_image_write.h"

// https://documentation.aimms.com/language-reference/procedural-language-components/external-procedures-and-functions/c-versus-fortran-conventions.html

// https://raytracing.github.io/books/RayTracingInOneWeekend.html
// Stop on `The vec3 Class`


color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + random_double(), 0.2, b + random_double());
            shared_ptr<material> sphere_material;

            if (choose_mat < 0.8) {
                // diffuse
                auto albedo = color::random() * color::random();
                sphere_material = make_shared<lambertian>(albedo);
                world.add(make_shared<sphere>(center, 0.2, sphere_material));
            } else if (choose_mat < 0.95) {
                // metal
                auto albedo = color::random(0.5, 1);
                auto fuzz = random_double(0, 0.5);
                sphere_material = make_shared<metal>(albedo, fuzz);
                world.add(make_shared<sphere>(center, 0.2, sphere_material));
            } else {
                // glass
                sphere_material = make_shared<dielectric>(1.5);
                world.add(make_shared<sphere>(center, 0.2, sphere_material));
            }
            
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}


hittable_list small_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}


int render_scene(unsigned char* img, const int image_height, const int image_width, const bool isBigScene) {
    // Image
    const int samples_per_pixel = 20;
    const int max_depth = 50;
    int aspect_ratio = image_height / image_width;

    // World
    auto R = cos(pi/4);
    hittable_list world;
    if (isBigScene) {
        world = random_scene();
    } else {
        world = small_scene();
    }
    /*
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left   = make_shared<dielectric>(1.5);
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0), -0.45, material_left));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
    */
    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
    // Render
    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < image_width; ++j) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; s++) {
                auto u = (j + random_double()) / (image_width-1);
                auto v = (i + random_double()) / (image_height-1);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(img, pixel_color, samples_per_pixel, i, j, image_height, image_width, 3);
        }
    }
}


int render_scene_parallel(unsigned char* img, const int image_height, const int image_width, const bool isBigScene) {
    // Image
    const int samples_per_pixel = 20;
    const int max_depth = 50;
    int aspect_ratio = image_height / image_width;

    // World
    hittable_list world;
    if (isBigScene) {
        world = random_scene();
    } else {
        world = small_scene();
    }
    /*
    auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left   = make_shared<dielectric>(1.5);
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<sphere>(point3( 0.0,    0.0, -1.0),   0.5, material_center));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<sphere>(point3(-1.0,    0.0, -1.0), -0.45, material_left));
    world.add(make_shared<sphere>(point3( 1.0,    0.0, -1.0),   0.5, material_right));
    */
    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    camera cam(lookfrom, lookat, vec3(0,1,0), 30, aspect_ratio, aperture, dist_to_focus);
    // Render
    omp_set_num_threads(6);
    int i, j, s;
    #pragma omp parallel shared(img)
    {
        #pragma omp for schedule(static, 50) collapse(2) private(i, j, s)
        for (i = 0; i < image_height; ++i) {
            for (j = 0; j < image_width; ++j) {
                color pixel_color(0, 0, 0);
                for (s = 0; s < samples_per_pixel; s++) {
                    auto u = (j + random_double()) / (image_width-1);
                    auto v = (i + random_double()) / (image_height-1);
                    ray r = cam.get_ray(u, v);
                    pixel_color += ray_color(r, world, max_depth);
                }
                write_color(img, pixel_color, samples_per_pixel, i, j, image_height, image_width, 3);
            }
        }
    }
}


int main() {
    // Image
    const int image_width = 256;
    const int image_height = 256;
    char const *filename = "image.bmp";
    unsigned char *img = (unsigned char*)malloc(image_height * image_width * 3 * sizeof(unsigned char));
    render_scene_parallel(img, image_height, image_width, false);
    stbi_write_bmp(
        filename, image_width, image_height, 3, 
        img
    );
}