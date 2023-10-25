#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#include "../../include/hex.h"



static const uint8_t *half2upper = "0123456789ABCDEF";
static const uint8_t *half2lower = "0123456789abcdef";

#ifdef USE_LUT
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
static uint8_t col2half(uint8_t c)
{
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
}
#endif


static int base16_encode_(size_t input_size, const uint8_t *src, uint8_t *dst, const uint8_t *table)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint8_t unit = src[i];

        uint8_t c0 = (unit >> 0) & 0x0F;
        uint8_t c1 = (unit >> 4) & 0x0F;

        c0 = table[c0];
        c1 = table[c1];

        dst[i * 2 + 0] = c1;
        dst[i * 2 + 1] = c0;
    }

    return 1;
}
static int base16_inplace_encode_(size_t input_size, uint8_t *data, const uint8_t *table)
{
    for (int i = units; i-- > 0;)
    {
        uint8_t unit = data[i];

        uint8_t c0 = (unit >> 0) & 0x0F;
        uint8_t c1 = (unit >> 4) & 0x0F;

        c0 = table[c0];
        c1 = table[c1];

        data[i * 2 + 0] = c1;
        data[i * 2 + 1] = c0;
    }

    return 1;
}
int base16_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_(input_size, src, dst, half2upper);
}
int base16_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_(input_size, src, dst, half2lower);
}
int base16_64n_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_u(input_size - input_size % 64, src, dst);
}
int base16_64n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base16_encode_l(input_size - input_size % 64, src, dst);
}
int base16_inplace_encode_u(size_t input_size, uint8_t *data)
{
    return base16_inplace_encode_(input_size, data, half2upper);
}
int base16_inplace_encode_l(size_t input_size, uint8_t *data)
{
    return base16_inplace_encode_(input_size, data, half2lower);
}



int base16_2n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    size_t units = input_size / 2;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint8_t c0 = src[i * 2];
        uint8_t c1 = src[i * 2 + 1];

        c0 = col2half(c0);
        c1 = col2half(c1);

        c0 = (c0 << 4) | c1;

        dst[i] = c0;
    }

    return 1;
}
int base16_2n_inplace_decode(size_t input_size, uint8_t *data)
{
    size_t units = input_size / 2;

    for (int i = 0; i < units; ++i)
    {
        uint8_t c0 = data[i * 2];
        uint8_t c1 = data[i * 2 + 1];

        c0 = col2half(c0);
        c1 = col2half(c1);

        c0 = (c0 << 4) | c1;

        data[i] = c0;
    }

    return 1;
}
int base16_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (!base16_2n_decode(input_size & ~1, src, dst)) return 0;

    if (input_size & 1)
    {
        uint8_t c = src[input_size - 1];
        c = col2half(c);

        dst[(input_size - 1) / 2 - 1] = c << 8;
    }

    return 1;
}
int base16_inplace_decode(size_t input_size, uint8_t *data)
{
    if (!base16_2n_inplace_decode(input_size & ~1, data)) return 0;

    if (input_size & 1)
    {
        uint8_t c = data[input_size - 1];
        c = col2half(c);

        data[(input_size - 1) / 2 - 1] = c << 8;
    }

    return 1;
}
