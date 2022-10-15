#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include <iostream>

void write_color(unsigned char *data, color pixel_color, int samples_per_pixel, int i, int j, int h, int w, int c = 3) {
    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel;
    data[0 + c * (j + w * i)] = static_cast<int>(
        256 * clamp(sqrt(pixel_color.x() * scale), 0.0, 0.999)
    );
    data[1 + c * (j + w * i)] = static_cast<int>(
        256 * clamp(sqrt(pixel_color.y() * scale), 0.0, 0.999)
    );
    data[2 + c * (j + w * i)] = static_cast<int>(
        256 * clamp(sqrt(pixel_color.z() * scale), 0.0, 0.999)
    );
}

#endif