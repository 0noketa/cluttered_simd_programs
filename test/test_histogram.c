#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"


#define DATA8_SIZE 256
alignas(32) int8_t data8_a[DATA8_SIZE * 4];
alignas(32) int8_t data8_b[DATA8_SIZE * 4];
alignas(32) int8_t data8_c[DATA8_SIZE * 4];

int main()
{
    for (int i = 0; i < 256; ++i)
    {
        data8_a[i] = INT8_MIN + i;
    }

    for (int i = 0; i < 64; ++i)
    {
        data8_b[i] = INT8_MIN + i / 2;
        data8_b[DATA8_SIZE - 1 - i] = INT8_MIN + i / 2;
    }
    for (int i = 0; i < 64; ++i)
    {
        data8_b[128 - i] = INT8_MAX - i * 3.5;
        data8_b[128 + i] = INT8_MAX - i * 3.5;
    }

    for (int i = 0; i < 256; ++i)
    {
        if (i % 23 == 0) data8_c[i] = -128;
        else if (i % 17 == 0) data8_c[i] = -96;
        else if (i % 13 == 0) data8_c[i] = -64;
        else if (i % 11 == 0) data8_c[i] = -32;
        else if (i % 7 == 0) data8_c[i] = 0;
        else if (i % 5 == 0) data8_c[i] = 32;
        else if (i % 3 == 0) data8_c[i] = 64;
        else if (i % 2 == 0) data8_c[i] = 96;
    }

    int8_t *data8s[3] = { data8_a, data8_b, data8_c };
    for (int i = 0; i < 3; ++i)
    {
        size_t result0 = 0;
        int8_t _min, _max;
        vec_i8x32n_get_minmax(DATA8_SIZE, data8s[i], &_min, &_max);

        puts("---- input ----");
        int unit = DATA8_SIZE / 16;
        int half_w = 16;
        for (int k = 0; k < DATA8_SIZE / unit; ++k)
        {
            int base = unit * k;
            int _sum = 0;
            for (int n = 0; n < unit; ++n)
            {
                _sum += data8s[i][base + n];
            }
            int it = _sum / unit;
            for (int j = -half_w; j <= half_w; ++j)
            {
                printf("%c", (it / ((INT8_MAX / half_w) + 1) == j ? '*' : j == 0 ? '|' : ' '));
            }
            puts("");
        }

        uint8_t histogram[16] = {0,};
        vec_i8x32n_get_histogram_u8x8(DATA8_SIZE, data8s[i], _min, _max, histogram);

        printf("%d - %d\n", (int)_min, (int)_max);
        puts("--- output ---");
        for (int i = 0; i < 8; ++i)
        { 
            printf("%.f\t- : %3d|", (float)(_min + (_max - _min) / 8 * i), (int)histogram[i]);
            for (int j = 0; j < histogram[i] / (256 / 32); ++j) putchar('#');
            puts("");
        }
        for (int i = 8; i < sizeof(histogram); ++i)
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
