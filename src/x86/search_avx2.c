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

#include "../include/simd_tools.h"


/* min/max */

size_t vec_i32v8n_get_min_index(size_t size, int32_t *src);
size_t vec_i32v8n_get_max_index(size_t size, int32_t *src);
void vec_i32v8n_get_minmax_index(size_t size, int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32v8n_get_min(size_t size, int32_t *src);
int32_t vec_i32v8n_get_max(size_t size, int32_t *src);
void vec_i32v8n_get_minmax(size_t size, int32_t *src, int32_t *out_min, int32_t *out_max);


size_t vec_i16v16n_get_min_index(size_t size, int16_t *src)
;
size_t vec_i16v16n_get_max_index(size_t size, int16_t *src)
;
void vec_i16v16n_get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
;

int16_t vec_i16v16n_get_min(size_t size, int16_t *src)
{
    size_t units = size / 16;
    __m256i *p = (__m256i*)src;

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

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 16;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

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

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 16;
    __m256i current_min;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

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


size_t vec_i8v32n_get_min_index(size_t size, int8_t *src)
;
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src)
;
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max)
;

int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 32;
    __m256i current_min;
    __m256i *p = (__m256i*)src;

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
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 32;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

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
void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 32;
    __m256i current_min;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

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

int32_t vec_i32v8n_count(size_t size, int32_t *src, int32_t element)
;
int16_t vec_i16v16n_count(size_t size, int16_t *src, int16_t element)
;
int8_t vec_i8v32n_count(size_t size, int8_t *src, int8_t element)
;
