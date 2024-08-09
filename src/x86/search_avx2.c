// #define _M_IX86
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#if defined(__GNUC__)
#include <x86intrin.h>
#elif defined(_MSC_VER) 
#include <intrin.h>
#endif

#include "../../include/search.h"


/* min/max */

size_t vec_i32x8n_get_min_index(size_t size, const int32_t *src);
size_t vec_i32x8n_get_max_index(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32x8n_get_min(size_t size, const int32_t *src);
int32_t vec_i32x8n_get_max(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max);


static void vec_i16x16n_get_min_and_index_i16_i(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_idx)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;

    __m128i iota = _mm_set_epi16(7, 6, 5, 4,  3, 2, 1, 0);
    __m128i eight_x8 = _mm_set1_epi16(8);
    __m128i max_x8 = _mm_set1_epi16(UINT16_MAX);
    __m128i current_min = _mm_set1_epi16(INT16_MAX);
    __m128i current_min_idx = _mm_set1_epi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i current_min0 = current_min;
        current_min = _mm_min_epi16(current_min, it);

        __m128i mask = _mm_cmpeq_epi16(current_min0, current_min);
        __m128i idx = _mm_andnot_si128(mask, iota);
        current_min_idx = _mm_and_si128(mask, current_min_idx);
        current_min_idx = _mm_or_si128(current_min_idx, idx);
        
        iota = _mm_adds_epi16(iota, eight_x8);
    }

    {
        __m128i current_min_hi = _mm_srli_si128(current_min, 8);
        __m128i current_min_idx_hi = _mm_srli_si128(current_min_idx, 8);

        __m128i current_min0 = current_min;
        current_min = _mm_min_epi16(current_min, current_min_hi);

        __m128i mask = _mm_cmpeq_epi16(current_min0, current_min);
        current_min_idx_hi = _mm_andnot_si128(mask, current_min_idx_hi);
        current_min_idx = _mm_and_si128(mask, current_min_idx);
        current_min_idx = _mm_or_si128(current_min_idx, current_min_idx_hi);
    }

    {
        __m128i current_min_hi = _mm_srli_si128(current_min, 4);
        __m128i current_min_idx_hi = _mm_srli_si128(current_min_idx, 4);

        __m128i current_min0 = current_min;
        current_min = _mm_min_epi16(current_min, current_min_hi);

        __m128i mask = _mm_cmpeq_epi16(current_min0, current_min);
        current_min_idx_hi = _mm_andnot_si128(mask, current_min_idx_hi);
        current_min_idx = _mm_and_si128(mask, current_min_idx);
        current_min_idx = _mm_or_si128(current_min_idx, current_min_idx_hi);
    }

    uint32_t results = _mm_cvtsi128_si32(current_min);
    uint32_t results_idx = _mm_cvtsi128_si32(current_min_idx);
    int16_t lo = results & 0xFFFF;
    int16_t lo_idx = results_idx & 0xFFFF;
    int16_t hi = (results >> 16) & 0xFFFF;
    int16_t hi_idx = (results_idx >> 16) & 0xFFFF;

    if (lo < hi)
    {
        *out_min = lo;
        *out_idx = lo_idx;
    }
    else
    {
        *out_min = hi;
        *out_idx = hi_idx;
    }
}
int16_t vec_i16x16n_get_min_index_i16(size_t size, const int16_t *src)
{
    int16_t val;
    int16_t result;
    vec_i16x16n_get_min_and_index_i16_i(size, src, &result, &val);
    return result;
}
size_t vec_i16x16n_get_min_index(size_t size, const int16_t *src)
{
    const size_t BLOCK_SIZE = 0x4000;
    size_t units = size / BLOCK_SIZE;
    int16_t current_min = INT16_MAX;
    size_t current_min_index = 0;

    for (int i = 0; i < units; ++i)
    {
        int16_t it;
        int16_t idx;
        vec_i16x16n_get_min_and_index_i16_i(BLOCK_SIZE, src + i * BLOCK_SIZE, &it, &idx);

        if (it < current_min)
        {
            current_min = it;
            current_min_index = i * BLOCK_SIZE + idx;
        }
    }

    return current_min_index;
}

static void vec_i16x16n_get_max_and_index_i16_i(size_t size, const int16_t *src, int16_t *out_max, int16_t *out_idx)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;

    __m128i iota = _mm_set_epi16(7, 6, 5, 4,  3, 2, 1, 0);
    __m128i eight_x8 = _mm_set1_epi16(8);
    __m128i max_x8 = _mm_set1_epi16(UINT16_MAX);
    __m128i current_max = _mm_set1_epi16(INT16_MIN);
    __m128i current_max_idx = _mm_set1_epi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i current_max0 = current_max;
        current_max = _mm_max_epi16(current_max, it);

        __m128i mask = _mm_cmpeq_epi16(current_max0, current_max);
        __m128i idx = _mm_andnot_si128(mask, iota);
        current_max_idx = _mm_and_si128(mask, current_max_idx);
        current_max_idx = _mm_or_si128(current_max_idx, idx);
        
        iota = _mm_adds_epi16(iota, eight_x8);
    }

    {
        __m128i current_max_hi = _mm_srli_si128(current_max, 8);
        __m128i current_max_idx_hi = _mm_srli_si128(current_max_idx, 8);

        __m128i current_max0 = current_max;
        current_max = _mm_max_epi16(current_max, current_max_hi);

        __m128i mask = _mm_cmpeq_epi16(current_max0, current_max);
        current_max_idx_hi = _mm_andnot_si128(mask, current_max_idx_hi);
        current_max_idx = _mm_and_si128(mask, current_max_idx);
        current_max_idx = _mm_or_si128(current_max_idx, current_max_idx_hi);
    }

    {
        __m128i current_max_hi = _mm_srli_si128(current_max, 4);
        __m128i current_max_idx_hi = _mm_srli_si128(current_max_idx, 4);

        __m128i current_max0 = current_max;
        current_max = _mm_max_epi16(current_max, current_max_hi);

        __m128i mask = _mm_cmpeq_epi16(current_max0, current_max);
        current_max_idx_hi = _mm_andnot_si128(mask, current_max_idx_hi);
        current_max_idx = _mm_and_si128(mask, current_max_idx);
        current_max_idx = _mm_or_si128(current_max_idx, current_max_idx_hi);
    }

    uint32_t results = _mm_cvtsi128_si32(current_max);
    uint32_t results_idx = _mm_cvtsi128_si32(current_max_idx);
    int16_t lo = results & 0xFFFF;
    int16_t lo_idx = results_idx & 0xFFFF;
    int16_t hi = (results >> 16) & 0xFFFF;
    int16_t hi_idx = (results_idx >> 16) & 0xFFFF;

    if (lo > hi)
    {
        *out_max = lo;
        *out_idx = lo_idx;
    }
    else
    {
        *out_max = hi;
        *out_idx = hi_idx;
    }
}
int16_t vec_i16x16n_get_max_index_i16(size_t size, const int16_t *src)
{
    int16_t val;
    int16_t result;
    vec_i16x16n_get_max_and_index_i16_i(size, src, &result, &val);
    return result;
}
size_t vec_i16x16n_get_max_index(size_t size, const int16_t *src)
{
    const size_t BLOCK_SIZE = 0x4000;
    size_t units = size / BLOCK_SIZE;
    int16_t current_max = INT16_MIN;
    size_t current_max_index = 0;

    for (int i = 0; i < units; ++i)
    {
        int16_t it;
        int16_t idx;
        vec_i16x16n_get_max_and_index_i16_i(BLOCK_SIZE, src + i * BLOCK_SIZE, &it, &idx);

        if (it > current_max)
        {
            current_max = it;
            current_max_index = i * BLOCK_SIZE + idx;
        }
    }

    return current_max_index;
}
void vec_i16x16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max)
;

int16_t vec_i16x16n_get_min(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    const __m256i *p = (const __m256i*)src;

    __m256i current_min = _mm256_set1_epi16(INT16_MAX);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_min = _mm256_min_epi16(current_min, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_min = _mm256_min_epi16(current_min, current_min_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    result_min0 = _mm_min_epi16(result_min0, result_min1);

    int16_t result = _mm_cvtsi128_si32(result_min0) & UINT16_MAX;
    return result;
}

int16_t vec_i16x16n_get_max(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    __m256i current_max;
    const __m256i *p = (const __m256i*)src;

    current_max = _mm256_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_max = _mm256_max_epi16(current_max, it);
    }

    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_max = _mm256_max_epi16(current_max, current_max_lo);

    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_max0 = _mm_max_epi16(result_max0, result_max1);

    int16_t result = _mm_cvtsi128_si32(result_max0) & UINT16_MAX;
    return result;
}

void vec_i16x16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 16;
    __m256i current_min;
    __m256i current_max;
    const __m256i *p = (const __m256i*)src;

    current_min = _mm256_set1_epi16(INT16_MAX);
    current_max = _mm256_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_min = _mm256_min_epi16(current_min, it);
        current_max = _mm256_max_epi16(current_max, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_min0 = _mm_min_epi16(result_min0, result_min1);
    result_max0 = _mm_max_epi16(result_max0, result_max1);

    *out_min = _mm_cvtsi128_si32(result_min0) & UINT16_MAX;
    *out_max = _mm_cvtsi128_si32(result_max0) & UINT16_MAX;
 }


size_t vec_i8x32n_get_min_index(size_t size, const int8_t *src)
;
size_t vec_i8x32n_get_max_index(size_t size, const int8_t *src)
;
void vec_i8x32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max)
;

int8_t vec_i8x32n_get_min(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    __m256i current_min;
    const __m256i *p = (const __m256i*)src;

    current_min = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_min = _mm256_min_epi8(current_min, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 1);
    current_min = _mm256_min_epi8(current_min, current_min_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    result_min0 = _mm_min_epi8(result_min0, result_min1);

    int8_t result = _mm_cvtsi128_si32(result_min0) & UINT8_MAX;
    return result;
}
int8_t vec_i8x32n_get_max(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    __m256i current_max;
    const __m256i *p = (const __m256i*)src;

    current_max = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_max = _mm256_max_epi8(current_max, it);
    }

    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 1);
    current_max = _mm256_max_epi8(current_max, current_max_lo);

    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_max0 = _mm_max_epi8(result_max0, result_max1);

    int8_t result = _mm_cvtsi128_si32(result_max0) & UINT8_MAX;
    return result;
}
/*stub*/
void vec_i8x32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 32;
    __m256i current_min;
    __m256i current_max;
    const __m256i *p = (const __m256i*)src;

    current_min = _mm256_set1_epi8(INT8_MAX);
    current_max = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_min = _mm256_min_epi8(current_min, it);
        current_max = _mm256_max_epi8(current_max, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 1);
    current_max_lo = _mm256_srli_si256(current_max, 1);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_min0 = _mm_min_epi8(result_min0, result_min1);
    result_max0 = _mm_max_epi8(result_max0, result_max1);

    *out_min = _mm_cvtsi128_si32(result_min0) & UINT8_MAX;
    *out_max = _mm_cvtsi128_si32(result_max0) & UINT8_MAX;
}


/* search */

int32_t vec_i32x8n_count_i32(size_t size, const int32_t *src, int32_t value)
;
size_t vec_i32x8n_count(size_t size, const int32_t *src, int32_t value)
;
static __m128i vec_i16x16n_count_m128(size_t size, const int16_t *src, int16_t value)
{
    size_t units = size / 8;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i one_x4 = _mm_set1_epi16(1);
    __m128i needle = _mm_set1_epi16(value);

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i mask = _mm_cmpeq_epi16(it, needle);
        __m128i d = _mm_and_si128(mask, one_x4);
        results = _mm_adds_epu16(results, d);
    }

    return results;
}
int16_t vec_i16x16n_count_i16(size_t size, const int16_t *src, int16_t value)
{
    __m128i results = vec_i16x16n_count_m128(size, src, value);
    results = _mm_adds_epu16(results, _mm_srli_si128(results, 2));
    results = _mm_adds_epu16(results, _mm_srli_si128(results, 4));
    results = _mm_adds_epu16(results, _mm_srli_si128(results, 8));
    uint16_t result = _mm_cvtsi128_si32(results) & 0xFFFF;

    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_i16x16n_count(size_t size, const int16_t *src, int16_t value)
{
    const size_t unit_size = 0x8000 * 8;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m128i results = vec_i16x16n_count_m128(unit_size, src + i * unit_size, value);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_and_si128(_mm_srli_si128(results, 2), mask_lower);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        results = _mm_add_epi32(results, _mm_srli_si128(results, 8));
        size_t result2 = _mm_cvtsi128_si32(results);

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m128i results = vec_i16x16n_count_m128(size2, src + base, value);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_and_si128(_mm_srli_si128(results, 2), mask_lower);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        results = _mm_add_epi32(results, _mm_srli_si128(results, 8));
        size_t result2 = _mm_cvtsi128_si32(results);

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    return result;
}
// returns u8x16
static __m128i vec_i8x32n_count_m128(size_t size, const int8_t *src, int8_t value)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i one_x8 = _mm_set1_epi8(1);
    __m128i needle = _mm_set1_epi8(value);

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i mask = _mm_cmpeq_epi8(it, needle);
        __m128i results0 = _mm_and_si128(mask, one_x8);
        results = _mm_adds_epu8(results, results0);
    }

    return results;
}
// stub
int8_t vec_i8x32n_count_i8(size_t size, const int8_t *src, int8_t value)
{
    __m128i results = vec_i8x32n_count_m128(size, src, value);
    results = _mm_adds_epu8(results, _mm_srli_si128(results, 1));
    results = _mm_adds_epu8(results, _mm_srli_si128(results, 2));
    results = _mm_adds_epu8(results, _mm_srli_si128(results, 4));
    results = _mm_adds_epu8(results, _mm_srli_si128(results, 8));
    size_t result = _mm_cvtsi128_si32(results) & 0xFFFF;

    return result > INT8_MAX ? INT8_MAX : result;
}
// stub
size_t vec_i8x32n_count(size_t size, const int8_t *src, int8_t value)
{
    const size_t unit_size = 0x80 * 16;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m128i results = vec_i8x32n_count_m128(unit_size, src + i * unit_size, value);
        const __m128i mask_lower = _mm_set1_epi16(0x00FF);
        __m128i results2 = _mm_and_si128(_mm_srli_si128(results, 1), mask_lower);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi16(results, results2);
        results = _mm_add_epi16(results, _mm_srli_si128(results, 2));
        results = _mm_add_epi16(results, _mm_srli_si128(results, 4));
        results = _mm_add_epi16(results, _mm_srli_si128(results, 8));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m128i results = vec_i8x32n_count_m128(size2, src + base, value);
        const __m128i mask_lower = _mm_set1_epi16(0x00FF);
        __m128i results2 = _mm_and_si128(_mm_srli_si128(results, 1), mask_lower);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi16(results, results2);
        results = _mm_add_epi16(results, _mm_srli_si128(results, 2));
        results = _mm_add_epi16(results, _mm_srli_si128(results, 4));
        results = _mm_add_epi16(results, _mm_srli_si128(results, 8));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    return result;
}
