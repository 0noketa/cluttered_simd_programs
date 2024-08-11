#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"


#define DATA_SIZE 1024
alignas(32) int16_t data_a[DATA_SIZE];
alignas(32) int16_t data_b[DATA_SIZE];
alignas(32) int16_t data_c[DATA_SIZE];

int main()
{
    for (int i = 0; i < 1024; ++i)
    {
        data_a[i] = INT16_MIN + i * 64;
    }

    for (int i = 0; i < 256; ++i)
    {
        data_b[i] = INT16_MIN + i;
        data_b[DATA_SIZE - 1 - i] = INT16_MIN + i * 2;
    }
    for (int i = 0; i < 256; ++i)
    {
        data_b[512 - i] = INT16_MAX - i * 255;
        data_b[512 + i] = INT16_MAX - i * 255;
    }

    for (int i = 0; i < 1024; ++i)
    {
        if (i % 23 == 0) data_c[i] = -128;
        else if (i % 17 == 0) data_c[i] = -96;
        else if (i % 13 == 0) data_c[i] = -64;
        else if (i % 11 == 0) data_c[i] = -32;
        else if (i % 7 == 0) data_c[i] = 0;
        else if (i % 5 == 0) data_c[i] = 32;
        else if (i % 3 == 0) data_c[i] = 64;
        else if (i % 2 == 0) data_c[i] = 127;
    }

    int16_t *datas[3] = { data_a, data_b, data_c };
    for (int i = 0; i < 3; ++i)
    {
        size_t result0 = 0;
        int16_t _min, _max;
        vec_i16x16n_get_minmax(DATA_SIZE, datas[i], &_min, &_max);

        puts("---- input ----");
        int unit = DATA_SIZE / 16;
        int half_w = 16;
        for (int k = 0; k < DATA_SIZE / unit; ++k)
        {
            int base = unit * k;
            int _sum = 0;
            for (int n = 0; n < unit; ++n)
            {
                _sum += datas[i][base + n];
            }
            int it = _sum / unit;
            for (int j = -half_w; j <= half_w; ++j)
            {
                printf("%c", (it / ((INT16_MAX / half_w) + 1) == j ? '*' : j == 0 ? '|' : ' '));
            }
            puts("");
        }

        int16_t histogram[16] = {0,};
        vec_i16x16n_get_histogram_i16x8(DATA_SIZE, datas[i], _min, _max, histogram);

        printf("%d - %d\n", (int)_min, (int)_max);
        puts("--- output ---");
        for (int i = 0; i < 8; ++i)
        { 
            printf("%.f\t- : %3d|", (float)(_min + (_max - _min) / 8 * i), (int)histogram[i]);
            for (int j = 0; j < histogram[i] / (DATA_SIZE / 64); ++j) putchar('#');
            puts("");
        }
        for (int i = 8; i < sizeof(histogram) / sizeof(int16_t); ++i)
        {
            if (histogram[i] != 0)
            {
                puts("output error!");
                break;
            }
        }

    }

    return 0;
}
