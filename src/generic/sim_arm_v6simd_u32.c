
#include <stdint.h>

typedef uint32_t uint8x4_t;
typedef uint32_t uint16x2_t;
typedef uint32_t int8x4_t;
typedef uint32_t int16x2_t;


// thread_local
static uint32_t GE = 0;


int16x2_t __sxtb16(int8x4_t xs)
{
    int8_t x0 = xs & 0xFF;
    int8_t x1 = (xs >> 8) & 0xFF;
    int8_t x2 = (xs >> 16) & 0xFF;
    int8_t x3 = xs >> 24;
    int16_t x0b = x0;
    int16_t x2b = x2;

    return x0b | ((uint32_t)x2b << 16);
}
// stub
int16x2_t __ssat16(int16x2_t xs, uint32_t n)
{
    int16_t hi = xs >> 16;
    int16_t lo = xs & 0xFFFF;
    
    uint32_t m = 1 << n;

    if (hi >= m) hi = m - 1;
    if (hi < 0 && -hi >= m) hi = -(m - 1);
    if (lo >= m) lo = m - 1;
    if (lo < 0 && -lo >= m) lo = -(m - 1);

    return ((uint32_t)hi << 16) | lo;
}
uint16x2_t __usat16(int16x2_t xs, uint32_t n)
{
    int16_t hi = xs >> 16;
    int16_t lo = xs & 0xFFFF;
    
    uint32_t m = 1 << n;

    if (hi >= m) hi = m - 1;
    if (hi < 0) hi = 0;
    if (lo >= m) lo = m - 1;
    if (lo < 0) lo = 0;

    return ((uint32_t)hi << 16) | lo;
}


int16x2_t __ssub16(int16x2_t xs, int16x2_t ys)
{
    int16_t xhi = xs >> 16;
    int16_t xlo = xs & 0xFFFF;
    int16_t yhi = ys >> 16;
    int16_t ylo = ys & 0xFFFF;

    int16_t zhi = xhi - yhi;
    int16_t zlo = xlo - ylo;

    GE = (zhi > xhi ? 0xFFFF0000 : 0) | (zlo > xlo ? 0x0000FFFF : 0);

    return ((uint16_t)zhi << 16) | zlo;
}
uint16x2_t __usub16(uint16x2_t xs, uint16x2_t ys)
{
    uint32_t xhi = xs >> 16;
    uint32_t xlo = xs & 0xFFFF;
    uint32_t yhi = ys >> 16;
    uint32_t ylo = ys & 0xFFFF;

    uint16_t zhi = xhi - yhi;
    uint16_t zlo = xlo - ylo;

    GE = (zhi <= xhi ? 0xFFFF0000 : 0) | (zlo <= xlo ? 0x0000FFFF : 0);

    return (zhi << 16) | zlo;
}

uint16x2_t __uadd16(uint16x2_t xs, uint16x2_t ys)
{
    uint32_t xhi = xs >> 16;
    uint32_t xlo = xs & 0xFFFF;
    uint32_t yhi = ys >> 16;
    uint32_t ylo = ys & 0xFFFF;

    uint16_t zhi = xhi + yhi;
    uint16_t zlo = xlo + ylo;

    GE = (zhi < xhi ? 0xFFFF0000 : 0) | (zlo < xlo ? 0x0000FFFF : 0);

    return (zhi << 16) | zlo;
}

uint16x2_t __sel(uint16x2_t xs, uint16x2_t ys)
{
    uint16x2_t z = (xs & GE) | (ys & ~GE);

    return z;
}

uint8x4_t __usub8(uint8x4_t xs, uint8x4_t ys)
{
    uint32_t x0 = xs & 0xFF;
    uint32_t x1 = (xs >> 8) & 0xFF;
    uint32_t x2 = (xs >> 16) & 0xFF;
    uint32_t x3 = xs >> 24;
    uint32_t y0 = ys & 0xFF;
    uint32_t y1 = (ys >> 8) & 0xFF;
    uint32_t y2 = (ys >> 16) & 0xFF;
    uint32_t y3 = ys >> 24;

    uint8_t z0 = x0 - y0;
    uint8_t z1 = x1 - y1;
    uint8_t z2 = x2 - y2;
    uint8_t z3 = x3 - y3;

    GE = (z3 <= x3 ? 0xFF000000 : 0)  | (z2 <= x2 ? 0x00FF0000 : 0) | (z1 <= x1 ? 0x0000FF00 : 0) | (z0 <= x0 ? 0x000000FF : 0);

    return (z3 << 24) | (z2 << 16) | (z1 << 8) | z0;
}

uint16x2_t __uadd8(uint16x2_t xs, uint16x2_t ys)
{
    uint32_t x0 = xs & 0xFF;
    uint32_t x1 = (xs >> 8) & 0xFF;
    uint32_t x2 = (xs >> 16) & 0xFF;
    uint32_t x3 = xs >> 24;
    uint32_t y0 = ys & 0xFF;
    uint32_t y1 = (ys >> 8) & 0xFF;
    uint32_t y2 = (ys >> 16) & 0xFF;
    uint32_t y3 = ys >> 24;

    uint8_t z0 = x0 + y0;
    uint8_t z1 = x1 + y1;
    uint8_t z2 = x2 + y2;
    uint8_t z3 = x3 + y3;

    GE = (z3 < x3 ? 0xFF000000 : 0)  | (z2 < x2 ? 0x00FF0000 : 0) | (z1 < x1 ? 0x0000FF00 : 0) | (z0 < x0 ? 0x000000FF : 0);

    return (z3 << 24) | (z2 << 16) | (z1 << 8) | z0;
}
