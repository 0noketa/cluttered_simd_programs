#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#include "../../include/hex.h"



#ifdef USE_LUT
static const uint32_t *half2upper = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'
};
static const uint32_t *half2lower = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
};

static uint8_t col2half_[256] = {
     0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0, 1, 2, 3, 4, 5, 6, 7,  8, 9, 0, 0, 0, 0, 0, 0,

     0,10,11,12,13,14,15, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0,10,11,12,13,14,15, 0,  0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,

     0,
};
#define col2half(n) col2half_[n]
#else
static const uint8_t *half2upper = "0123456789ABCDEF";
static const uint8_t *half2lower = "0123456789abcdef";

static uint32_t col2half(uint32_t c)
{
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
}
#endif


#ifdef USE_LUT
static int base16_4n_encode_(size_t input_size, const uint8_t *src, uint8_t *dst, const uint32_t *table)
#else
static int base16_4n_encode_(size_t input_size, const uint8_t *src, uint8_t *dst, const uint8_t *table)
#endif
{
    int units = input_size / 4;

    const uint32_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint32_t unit = p[i];

        uint32_t c0 = (unit >> 0) & 0x0F;
        uint32_t c1 = (unit >> 4) & 0x0F;
        uint32_t c2 = (unit >> 8) & 0x0F;
        uint32_t c3 = (unit >> 12) & 0x0F;
        uint32_t c4 = (unit >> 16) & 0x0F;
        uint32_t c5 = (unit >> 20) & 0x0F;
        uint32_t c6 = (unit >> 24) & 0x0F;
        uint32_t c7 = (unit >> 28);

        c0 = table[c0];
        c1 = table[c1];
        c2 = table[c2];
        c3 = table[c3];

        c4 = table[c4];
        c5 = table[c5];
        c6 = table[c6];
        c7 = table[c7];

        q[i * 2 + 0] = (c1 << 0) | (c0 << 8) | (c3 << 16) | (c2 << 24);
        q[i * 2 + 1] = (c5 << 0) | (c4 << 8) | (c7 << 16) | (c6 << 24);
    }

    return 1;
}
int base16_64n_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_4n_encode_u(input_size - input_size % 64, src, dst);
}
int base16_64n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_4n_encode_l(input_size - input_size % 64, src, dst);
}
int base16_4n_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_4n_encode_(input_size, src, dst, half2upper);
}
int base16_4n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_4n_encode_(input_size, src, dst, half2lower);
}
static int base16_encode_(size_t input_size, const uint8_t *src, uint8_t *dst, const uint8_t *table)
{
    int base = input_size - input_size % 4;
    if (!base16_4n_encode_(input_size, src, dst, table)) return 0;

    int units = input_size % 4;

    const uint32_t *p0 = (void*)src;
    const uint32_t *p = p0 + base;
    uint32_t *q0 = (void*)dst;
    uint32_t *q = q0 + base * 2;

    for (int i = 0; i < units; ++i)
    {
        uint8_t unit = p[i];

        uint32_t c0 = (unit >> 0) & 0x0F;
        uint32_t c1 = (unit >> 4) & 0x0F;

        c0 = table[c0];
        c1 = table[c1];

        q[i * 2 + 0] = c1;
        q[i * 2 + 1] = c0;
    }

    return 1;
}
int base16_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_with_table(input_size, src, dst, half2upper);
}
int base16_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_with_table(input_size, src, dst, half2lower);
}


// input size: 128n bytes (1024n bits)
// output size: a half of input size
int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_8n_decode(input_size - input_size % 128, src, dst);
}
int base16_8n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    // 8n bytes as uint32x8 x 2n per step
    size_t units = input_size / 2 / 4;

    const uint32_t *p = (void*)src;
    uint32_t *q = (void*)dst;

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
int base16_2n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    size_t base = input_size - input_size % 8;
    if (!base16_128n_decode(base, src, dst)) return 0;

    size_t units = (input_size % 8) / 2;

    for (int i = 0; i < units; ++i)
    {
        uint32_t c0 = src[base + i * 2];
        uint32_t c1 = src[base + i * 2 + 1];

        c0 = col2half[c0];
        c1 = col2half[c1];

        c0 = (c0 << 4) | c1;

        dst[base / 2 + i] = c0;
    }

    return 1;
}
int base16_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (!base16_2n_decode(input_size & ~1, src, dst)) return 0;

    if (input_size & 1)
    {
        uint32_t c = src[input_size - 1];
        c = col2half[c];

        dst[(input_size - 1) / 2 - 1] = c << 8;
    }

    return 1;
}
