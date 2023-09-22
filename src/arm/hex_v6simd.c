#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>

#if defined(__arm__)
#include <arm_acle.h>
#else
typedef uint32_t uint8x4_t;
typedef uint32_t uint16x2_t;
typedef uint32_t int8x4_t;
typedef uint32_t int16x2_t;
extern int16x2_t __ssat16(int16x2_t xs, uint32_t n);
extern int16x2_t __ssub16(int16x2_t xs, int16x2_t ys);
extern int16x2_t __sadd16(int16x2_t xs, int16x2_t ys);
extern int8x4_t __ssub8(int8x4_t xs, int8x4_t ys);
extern int8x4_t __sadd8(int8x4_t xs, int8x4_t ys);
extern int16x2_t __sxtb16(int8x4_t xs);
extern uint16x2_t __usat16(int16x2_t xs, uint32_t n);
extern uint16x2_t __usub16(uint16x2_t xs, uint16x2_t ys);
extern uint16x2_t __uadd16(uint16x2_t xs, uint16x2_t ys);
extern uint8x4_t __usub8(uint8x4_t xs, uint8x4_t ys);
extern uint8x4_t __uadd8(uint8x4_t xs, uint8x4_t ys);
extern uint16x2_t __sel(uint16x2_t xs, uint16x2_t ys);
#endif

#include "../../include/hex.h"


static const uint8_t *half2upper = "0123456789ABCDEF";
static const uint8_t *half2lower = "0123456789abcdef";

static uint32_t col2half(uint32_t c)
{
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
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
    int units = input_size / 4;

    const uint8x4_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint8x4_t unit = p[i];
        uint8x4_t unit_lo = (unit >> 4) & 0x0F0F0F0F;
        uint8x4_t unit_hi = unit & 0x0F0F0F0F;

        __usub8(unit_hi, 0x0A0A0A0A);
        unit_hi = __uadd8(unit_hi, __sel(0x37373737, 0x30303030));
        __usub8(unit_lo, 0x0A0A0A0A);
        unit_lo = __uadd8(unit_lo, __sel(0x37373737, 0x30303030));

        uint8x4_t unit_lo2 = ((unit_hi & 0x00FF00FF) << 8) | (unit_lo & 0x00FF00FF);
        uint8x4_t unit_hi2 = (unit_hi & 0xFF00FF00) | ((unit_lo & 0xFF00FF00) >> 8);

        q[i * 2 + 0] = (unit_lo2 & 0x0000FFFF) | ((unit_hi2 & 0x0000FFFF) << 16);
        q[i * 2 + 1] = ((unit_lo2 & 0xFFFF0000) >> 16) | (unit_hi2 & 0xFFFF0000);
    }

    return 1;
}
int base16_4n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    int units = input_size / 4;

    const uint8x4_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint8x4_t unit = p[i];
        uint8x4_t unit_lo = (unit >> 4) & 0x0F0F0F0F;
        uint8x4_t unit_hi = unit & 0x0F0F0F0F;

        __usub8(unit_hi, 0x0A0A0A0A);
        unit_hi = __uadd8(unit_hi, __sel(0x57575757, 0x30303030));
        __usub8(unit_lo, 0x0A0A0A0A);
        unit_lo = __uadd8(unit_lo, __sel(0x57575757, 0x30303030));

        uint8x4_t unit_lo2 = ((unit_hi & 0x00FF00FF) << 8) | (unit_lo & 0x00FF00FF);
        uint8x4_t unit_hi2 = (unit_hi & 0xFF00FF00) | ((unit_lo & 0xFF00FF00) >> 8);

        q[i * 2 + 0] = (unit_lo2 & 0x0000FFFF) | ((unit_hi2 & 0x0000FFFF) << 16);
        q[i * 2 + 1] = ((unit_lo2 & 0xFFFF0000) >> 16) | (unit_hi2 & 0xFFFF0000);
    }

    return 1;
}
int base16_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    int base = input_size - input_size % 4;
    if (!base16_4n_encode_u(input_size, src, dst)) return 0;

    int units = input_size % 4;

    const uint32_t *p0 = (void*)src + base;
    const uint32_t *p = p0 + base;
    uint32_t *q0 = (void*)dst;
    uint32_t *q = q0 + base * 2;

    for (int i = 0; i < units; ++i)
    {
        uint8_t unit = p[i];

        uint32_t c0 = (unit >> 0) & 0x0F;
        uint32_t c1 = (unit >> 4) & 0x0F;

        c0 = half2upper[c0];
        c1 = half2upper[c1];

        q[i * 2 + 0] = c1;
        q[i * 2 + 1] = c0;
    }

    return 1;
}
int base16_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    int base = input_size - input_size % 4;
    if (!base16_4n_encode_l(input_size, src, dst)) return 0;

    int units = input_size % 4;

    const uint32_t *p0 = (void*)src + base;
    const uint32_t *p = p0 + base;
    uint32_t *q0 = (void*)dst;
    uint32_t *q = q0 + base * 2;

    for (int i = 0; i < units; ++i)
    {
        uint8_t unit = p[i];

        uint32_t c0 = (unit >> 0) & 0x0F;
        uint32_t c1 = (unit >> 4) & 0x0F;

        c0 = half2lower[c0];
        c1 = half2lower[c1];

        q[i * 2 + 0] = c1;
        q[i * 2 + 1] = c0;
    }

    return 1;
}


// input size: 128n bytes (1024n bits)
// output size: a half of input size
int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
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
    size_t base = input_size - input_size % 128;
    if (!base16_128n_decode(base, src, dst)) return 0;

    size_t units = (input_size % 128) / 2;

    for (int i = 0; i < units; ++i)
    {
        uint32_t c0 = src[base + i * 2];
        uint32_t c1 = src[base + i * 2 + 1];

        c0 = col2half(c0);
        c1 = col2half(c1);

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
        c = col2half(c);

        dst[(input_size - 1) / 2 - 1] = c << 8;
    }

    return 1;
}
