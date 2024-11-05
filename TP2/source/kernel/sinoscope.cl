#include "helpers.cl"

typedef struct sinoscope_int {
    unsigned int buffer_size;
    unsigned int width;
    unsigned int height;
    unsigned int taylor;
    unsigned int interval;
} sinoscope_int_t;

typedef struct sinoscope_float {
    float interval_inverse;
    float time;
    float max;
    float phase0;
    float phase1;
    float dx;
    float dy;
} sinoscope_float_t;

__kernel void sinoscope_kernel(__global unsigned char *buffer, const sinoscope_int_t sinoscope_int, const sinoscope_float_t sinoscope_float) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float px    = sinoscope_float.dx * y - 2 * M_PI;
    float py    = sinoscope_float.dy * x - 2 * M_PI;
    float value = 0;

    for (int k = 1; k <= sinoscope_int.taylor; k += 2) {
        value += sin(px * k * sinoscope_float.phase1 + sinoscope_float.time) / k;
        value += cos(py * k * sinoscope_float.phase0) / k;
    }

    value = (atan(value) - atan(-value)) / M_PI;
    value = (value + 1) * 100;

    pixel_t pixel;
    color_value(&pixel, value, sinoscope_int.interval, sinoscope_float.interval_inverse);

    const int index = (x * 3) + (y * 3) * sinoscope_int.width;
    buffer[index] = pixel.bytes[0];
    buffer[index + 1] = pixel.bytes[1];
    buffer[index + 2] = pixel.bytes[2];
}