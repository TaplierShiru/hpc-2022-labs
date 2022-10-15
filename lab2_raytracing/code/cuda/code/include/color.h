#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

void write_color(unsigned char *data, vec3 pixel_color, int samples_per_pixel, int i, int j, int h, int w, int c = 3) {
    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel;
    data[0 + c * (j + w * i)] = static_cast<unsigned char>(
        256 * clamp(pixel_color.x(), 0.0, 0.999)
    );
    data[1 + c * (j + w * i)] = static_cast<unsigned char>(
        256 * clamp(pixel_color.y(), 0.0, 0.999)
    );
    data[2 + c * (j + w * i)] = static_cast<unsigned char>(
        256 * clamp(pixel_color.z(), 0.0, 0.999)
    );
}

#endif