#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>



static uint32_t col2half(uint32_t c)
{
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
}


// input size: 128n bytes (1024n bits)
// output size: a half of input size
int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    // 8n bytes as uint32x8 x 2n per step
    size_t units = input_size / 2 / 4;

    const uint32_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    // for (size_t i = 0; i < units; ++i)
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint32_t unit = p[i * 2];
        uint32_t unit2 = p[i * 2 + 1];

        uint32_t c0 = (unit >> 0) & 0xFF;
        uint32_t c1 = (unit >> 8) & 0xFF;
        uint32_t c2 = (unit >> 16) & 0xFF;
        uint32_t c3 = (unit >> 24) & 0xFF;

        uint32_t c4 = (unit2 >> 0) & 0xFF;
        uint32_t c5 = (unit2 >> 8) & 0xFF;
        uint32_t c6 = (unit2 >> 16) & 0xFF;
        uint32_t c7 = (unit2 >> 24) & 0xFF;

        c0 = col2half(c0);
        c1 = col2half(c1);
        c2 = col2half(c2);
        c3 = col2half(c3);

        c4 = col2half(c4);
        c5 = col2half(c5);
        c6 = col2half(c6);
        c7 = col2half(c7);

        c0 = (c0 << 4) | c1;
        c2 = (c2 << 4) | c3;
        c4 = (c4 << 4) | c5;
        c6 = (c6 << 4) | c7;

        unit = (c0 << 0) | (c2 << 8) | (c4 << 16) | (c6 << 24);
 
        q[i] = unit;
    }

    return 1;
}

