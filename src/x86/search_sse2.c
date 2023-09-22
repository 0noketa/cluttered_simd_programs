#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>

#include "../../include/search.h"


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
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_min = _mm_set1_epi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_min_epi16(current_min, it);
    }

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_min = _mm_min_epi16(current_min, current_min_lo);

    int16_t result = _mm_cvtsi128_si32(current_min) & UINT16_MAX;
    return result;
}

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_max = _mm_set1_epi16(INT16_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_max = _mm_max_epi16(current_max, it);    }

    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_max = _mm_max_epi16(current_max, current_max_lo);

    int16_t result = _mm_cvtsi128_si32(current_max) & UINT16_MAX;
    return result;
}

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_min = _mm_set1_epi16(INT16_MAX);
    __m128i current_max = _mm_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_min_epi16(current_min, it);
        current_max = _mm_max_epi16(current_max, it);
    }

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);

    *out_min = _mm_cvtsi128_si32(current_min) & UINT16_MAX;
    *out_max = _mm_cvtsi128_si32(current_max) & UINT16_MAX;
}


size_t vec_i8v32n_get_min_index(size_t size, int8_t *src)
;
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src)
;
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max)
;

int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_min = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_add_epi8(current_min, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_min = _mm_min_epu8(current_min, it);
        current_min = _mm_sub_epi8(current_min, diff_x_16);
    }

	current_min = _mm_add_epi8(current_min, diff_x_16);

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 1);
    current_min = _mm_min_epu8(current_min, current_min_lo);

	current_min = _mm_sub_epi8(current_min, diff_x_16);

    int8_t result = _mm_cvtsi128_si32(current_min) & UINT8_MAX;
    return result;
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_max = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_max = _mm_add_epi8(current_max, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_max = _mm_max_epu8(current_max, it);
        current_max = _mm_sub_epi8(current_max, diff_x_16);
    }

	current_max = _mm_add_epi8(current_max, diff_x_16);

    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 1);
    current_max = _mm_max_epu8(current_max, current_max_lo);

	current_max = _mm_sub_epi8(current_max, diff_x_16);

    int8_t result = _mm_cvtsi128_si32(current_max) & UINT8_MAX;
    return result;
}

void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_min = _mm_set1_epi8(INT8_MAX);
    __m128i current_max = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_add_epi8(current_min, diff_x_16);
        current_max = _mm_add_epi8(current_max, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_min = _mm_min_epu8(current_min, it);
        current_max = _mm_max_epu8(current_max, it);
        current_min = _mm_sub_epi8(current_min, diff_x_16);
        current_max = _mm_sub_epi8(current_max, diff_x_16);
    }

	current_min = _mm_add_epi8(current_min, diff_x_16);
	current_max = _mm_add_epi8(current_max, diff_x_16);

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 1);
    current_max_lo = _mm_srli_si128(current_max, 1);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);

	current_min = _mm_sub_epi8(current_min, diff_x_16);
	current_max = _mm_sub_epi8(current_max, diff_x_16);

    *out_min = _mm_cvtsi128_si32(current_min) & UINT8_MAX;
    *out_max  = _mm_cvtsi128_si32(current_max) & UINT8_MAX;
}


/* search */

int32_t vec_i32v8n_count_i32(size_t size, int32_t *src, int32_t value)
;
size_t vec_i32v8n_count(size_t size, int32_t *src, int32_t value)
;
int16_t vec_i16v16n_count_i16(size_t size, int16_t *src, int16_t value)
;
size_t vec_i16v16n_count(size_t size, int16_t *src, int16_t value)
;
int8_t vec_i8v32n_count_i8(size_t size, int8_t *src, int8_t value)
;
size_t vec_i8v32n_count(size_t size, int8_t *src, int8_t value)
;
