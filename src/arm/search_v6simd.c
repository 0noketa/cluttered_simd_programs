// ARMv6 SIMD
#include <stddef.h>
#include <stdint.h>

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

#include "../../include/search.h"


/* min/max */

size_t vec_i32x8n_get_min_index(size_t size, const int32_t *src);
size_t vec_i32x8n_get_max_index(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32x8n_get_min(size_t size, const int32_t *src);
int32_t vec_i32x8n_get_max(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max);


size_t vec_i16x16n_get_min_index(size_t size, const int16_t *src)
{
    int16_t current = INT16_MAX;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it < current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
size_t vec_i16x16n_get_max_index(size_t size, const int16_t *src)
{
    int16_t current = 0;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it > current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
void vec_i16x16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max)
{
    int16_t current_min = INT16_MAX;
    int16_t current_max = 0;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it < current_min)
        {
            current_min = it;
            current_min_index = i;
        } else if (it > current_max)
        {
            current_max = it;
            current_max_index = i;
        }

    }

    *out_min = current_min_index;
    *out_max = current_max_index;
}

int16_t vec_i16x16n_get_min(size_t size, const int16_t *src)
{
    return src[vec_i16x16n_get_min_index(size, src)];
}

int16_t vec_i16x16n_get_max(size_t size, const int16_t *src)
{
    return src[vec_i16x16n_get_max_index(size, src)];
}

void vec_i16x16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t min_idx, max_idx;
    vec_i16x16n_get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}


size_t vec_i8x32n_get_min_index(size_t size, const int8_t *src)
{
    int8_t current = INT8_MAX;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it < current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
size_t vec_i8x32n_get_max_index(size_t size, const int8_t *src)
{
    int8_t current = INT8_MIN;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it > current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
void vec_i8x32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max)
{
    int8_t current_min = INT8_MAX;
    int8_t current_max = INT8_MIN;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it < current_min)
        {
            current_min = it;
            current_min_index = i;
        } else if (it > current_max)
        {
            current_max = it;
            current_max_index = i;
        }

    }

    *out_min = current_min_index;
    *out_max = current_max_index;
}

// stub
int8_t vec_i8x32n_get_min(size_t size, const int8_t *src)
{
    int units = size / 4;
    const int8x4_t *p = (const void*)src;

    int16x2_t current = 0x7FFF7FFF;

    for (int i = 0; i < units; ++i)
    {
        int8x4_t it = p[i];
        int16x2_t it_hi = __sxtb16((it >> 8) & 0x00FF00FF);
        int16x2_t it_lo = __sxtb16(it & 0x00FF00FF);

        __usub16(0, __usub16(it_lo, it_hi));
        it_lo = __sel(it_lo, it_lo);
        __usub16(0, __usub16(it_lo, current));
        current = __sel(it_lo, current);
    }

    uint32_t results = current;
    int8_t v0 = results & 0xFFFF;
    int8_t v1 = (results >> 16) & 0xFFFF;
    return v0 < v1 ? v0 : v1;
}
// stub
int8_t vec_i8x32n_get_max(size_t size, const int8_t *src)
{
    int units = size / 4;
    const int8x4_t *p = (const void*)src;

    int16x2_t current = 0x80808080;

    for (int i = 0; i < units; ++i)
    {
        int8x4_t it = p[i];
        int16x2_t it_hi = __sxtb16((it >> 8) & 0x00FF00FF);
        int16x2_t it_lo = __sxtb16(it & 0x00FF00FF);

        __ssub16(0, __ssub16(it_lo, it_hi));
        it_lo = __sel(it_hi, it_lo);
        __ssub16(0, __ssub16(current, it_lo));
        current = __sel(it_lo, current);
    }

    uint32_t results = current;
    int8_t v0 = results & 0xFFFF;
    int8_t v1 = (results >> 16) & 0xFFFF;
    return v0 > v1 ? v0 : v1;
}
//stub
void vec_i8x32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
{
    int units = size / 4;
    const int8x4_t *p = (const void*)src;

    int8x4_t current_min = 0x7F7F7F7F;
    int8x4_t current_max = 0x80808080;

    for (int i = 0; i < units; ++i)
    {
        int8x4_t it = p[i];

        __ssub16(0, __ssub16(it, current_min));
        current_min = __sel(it, current_min);
        __ssub16(0, __ssub16(current_max, it));
        current_max = __sel(it, current_max);
    }

    uint32_t results = current_min;
    int8_t v0 = results & 0xFF;
    int8_t v1 = (results >> 8) & 0xFF;
    int8_t v2 = (results >> 16) & 0xFF;
    int8_t v3 = (results >> 24) & 0xFF;

    v0 = v0 < v1 ? v0 : v1;
    v2 = v2 < v3 ? v2 : v3;
    *out_min = v0 < v2 ? v0 : v2;

    results = current_max;
    v0 = results & 0xFF;
    v1 = (results >> 8) & 0xFF;
    v2 = (results >> 16) & 0xFF;
    v3 = (results >> 24) & 0xFF;

    v0 = v0 > v1 ? v0 : v1;
    v2 = v2 > v3 ? v2 : v3;
    *out_max = v0 > v2 ? v0 : v2;
}


//stub
uint8_t vec_u8v32n_get_min(size_t size, const uint8_t *src)
{
    int units = size / 4;
    const uint8x4_t *p = (const void*)src;

    uint8x4_t current = 0xFFFFFFFF;

    for (int i = 0; i < units; ++i)
    {
        uint8x4_t it = p[i];

        __usub8(it, current);
        current = __sel(it, current);
    }

    uint32_t results = current;
    uint8_t v0 = results & 0xFF;
    uint8_t v1 = (results >> 8) & 0xFF;
    uint8_t v2 = (results >> 16) & 0xFF;
    uint8_t v3 = (results >> 24) & 0xFF;

    v0 = v0 < v1 ? v0 : v1;
    v2 = v2 < v3 ? v2 : v3;
    return v0 < v2 ? v0 : v2;
}
uint8_t vec_u8v32n_get_max(size_t size, const uint8_t *src);


/* search */

int32_t vec_i32x8n_count_i32(size_t size, const int32_t *src, int32_t value)
{
    size_t result = vec_i32x8n_count(size, src, value);
    return result > INT32_MAX ? INT32_MAX : result;
}
size_t vec_i32x8n_count(size_t size, const int32_t *src, int32_t value)
{
    size_t result = 0;

    for (size_t i = 0; i < size; ++i)
    {
        result += (src[i] == value);
    }

    return result;
}

int16_t vec_i16x16n_count_i16(size_t size, const int16_t *src, int16_t value)
{
    size_t result = vec_i16x16n_count(size, src, value);
    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_i16x16n_count(size_t size, const int16_t *src, int16_t value)
{
    int size2 = size > 0x10000 ? 0x10000 : size;
    int units = size2 / 2;
    uint16x2_t *p0 = (void*)src;
    uint16x2_t needle = (uint16_t)value | ((uint16_t)value << 16);

    size_t result = 0;

    for (int i = 0; i < size / size2; ++i)
    {
        uint16x2_t *p = p0 + i * units;
        uint16x2_t results = (unsigned)units | ((unsigned)units << 16);

        for (int j = 0; j < units; ++j)
        {
            uint16x2_t ds = __usub16(needle, p[j]);
            // usat16 takes "signed" integers returns unsigned integers
            ds = (ds & 0x00FF00FF) | ((ds & 0xFF00FF00) >> 8);
            uint16x2_t is_neq = __usat16(ds, 1);
            results = __usub16(results, is_neq);
        }

        uint32_t result0 = ((uint32_t)results & 0xFFFF) + ((uint32_t)results >> 16);
        result += result0;
    }
    
    return result;
}

int8_t vec_i8x32n_count_i8(size_t size, const int8_t *src, int8_t value)
{
    size_t result = vec_i8x32n_count(size, src, value);
    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_i8x32n_count(size_t size, const int8_t *src, int8_t value)
{
    int size2 = size > 0x200 ? 0x200 : size;
    int units = size2 / 4;
    uint8x4_t *p0 = (void*)src;
    uint8x4_t needle = (uint8_t)value | ((uint8_t)value << 8) | ((uint8_t)value << 16) | ((uint8_t)value << 24);

    size_t result = 0;

    for (int i = 0; i < size / size2; ++i)
    {
        uint8x4_t *p = p0 + i * units;
        uint8x4_t results = (unsigned)units | ((unsigned)units << 8) | ((unsigned)units << 16) | ((unsigned)units << 24);

        for (int j = 0; j < units; ++j)
        {
            uint8x4_t ds = __usub8(needle, p[j]);
            uint8x4_t ds_lo = ds & 0x00FF00FF;
            uint8x4_t ds_hi = (ds >> 8) & 0x00FF00FF;
            uint8x4_t is_neq_lo = __usat16(ds_lo, 1);
            uint8x4_t is_neq_hi = __usat16(ds_hi, 1);
            uint8x4_t is_neq = (is_neq_hi << 8) | is_neq_lo;
            results = __usub8(results, is_neq);
    // printf("needle: %08X  vs: %08X  ds: %08X  neq: %08X  r: %08X\n", needle, p[j], ds, is_neq, results);
        }

        uint32_t result0 = ((uint32_t)results & 0xFF) + (((uint32_t)results >> 8) & 0xFF) + (((uint32_t)results >> 16) & 0xFF) + ((uint32_t)results >> 24);
        result += result0;
    }

    return result;
}




int32_t vec_i32x8n_get_index(size_t size, const int32_t *src, int32_t element);
int16_t vec_i16x16n_get_index(size_t size, const int16_t *src, int16_t element);
int8_t vec_i8x32n_get_index(size_t size, const int8_t *src, int8_t element);

int32_t vec_i32x8n_get_first_index(size_t size, const int32_t *src, int32_t element)
{
    size_t size2 = size > INT32_MAX ? INT32_MAX : size;

    for (size_t i = 0; i < size2; ++i)
    {
        if (src[i] == element) return i;
    }

    return size2;
}
int16_t vec_i16x16n_get_first_index(size_t size, const int16_t *src, int16_t element);
int8_t vec_i8x32n_get_first_index(size_t size, const int8_t *src, int8_t element);
