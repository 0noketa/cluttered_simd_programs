#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#include "../../include/search.h"

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif

#ifndef USE_ANDNOT
#define USE_ANDNOT
#endif


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
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    // ANY_EMMS();
    __m64 max_x_4 = _mm_set1_pi16(UINT16_MAX);
    __m64 current = _mm_set1_pi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(current, it);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x_4);
        current = _mm_and_si64(current, mask2);
#endif
        it = _mm_and_si64(it, mask);
        current = _mm_or_si64(current, it);
    }

    __m64 current2 = _mm_or_si64(_m_psllqi(current, 16), _m_psrlqi(current, 48));
    __m64 current3 = _mm_or_si64(_m_psllqi(current, 32), _m_psrlqi(current, 32));
    __m64 current4 = _mm_or_si64(_m_psllqi(current, 48), _m_psrlqi(current, 16));
    __m64 mask = _mm_cmpgt_pi16(current, current2);
    __m64 mask2 = _mm_cmpgt_pi16(current, current3);
    __m64 mask3 = _mm_cmpgt_pi16(current, current4);
    mask = _mm_xor_si64(mask, max_x_4);
#ifdef USE_ANDNOT
    mask = _mm_andnot_si64(mask2, mask);
    mask = _mm_andnot_si64(mask3, mask);
#else
    mask2 = _mm_xor_si64(mask2, max_x_4);
    mask3 = _mm_xor_si64(mask3, max_x_4);

    mask = _mm_and_si64(mask, mask2);
    mask = _mm_and_si64(mask, mask3);
#endif

    current = _mm_and_si64(current, mask);

    current = _mm_or_si64(current, _m_psrlqi(current, 32));
    current = _mm_or_si64(current, _m_psrlqi(current, 16));

    int16_t result = _m_to_int(current) & UINT16_MAX;

    ANY_EMMS();
    return result;
}

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
#ifdef USE_ANDNOT
    __m64 max_x_4 = _mm_set1_pi16(UINT16_MAX);
#endif
    __m64 current = _mm_set1_pi16(INT16_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(it, current);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x_4);
        current = _mm_and_si64(current, mask2);
#endif
        mask = _mm_and_si64(mask, it);
        current = _mm_or_si64(current, mask);
    }

    __m64 current2 = _mm_or_si64(_m_psllqi(current, 16), _m_psrlqi(current, 48));
    __m64 current3 = _mm_or_si64(_m_psllqi(current, 32), _m_psrlqi(current, 32));
    __m64 current4 = _mm_or_si64(_m_psllqi(current, 48), _m_psrlqi(current, 16));
    __m64 mask = _mm_cmpgt_pi16(current, current2);
    __m64 mask2 = _mm_cmpgt_pi16(current, current3);
    __m64 mask3 = _mm_cmpgt_pi16(current, current4);

    mask = _mm_and_si64(mask, mask2);
    mask = _mm_and_si64(mask, mask3);

    current = _mm_and_si64(current, mask);

    current = _mm_or_si64(current, _m_psrlqi(current, 32));
    current = _mm_or_si64(current, _m_psrlqi(current, 16));

    int16_t result = _m_to_int(current) & UINT16_MAX;

    ANY_EMMS();
    return result;
}

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    __m64 max_x_4 = _mm_set1_pi16(UINT16_MAX);
    __m64 current_min = _mm_set1_pi16(INT16_MAX);
    __m64 current_max = _mm_set1_pi16(INT16_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(current_min, it);
        __m64 mask_max = _mm_cmpgt_pi16(it, current_max);
#ifdef USE_ANFNOT
        // slow version
        current_min = _mm_andnot_si64(mask, current_min);
        current_max = _mm_andnot_si64(mask_max, current_max);
#else
        // fast version
        __m64 mask2 = _mm_xor_si64(mask, max_x_4);
        __m64 mask_max2 = _mm_xor_si64(mask_max, max_x_4);

        current_min = _mm_and_si64(current_min, mask2);
        current_max = _mm_and_si64(current_max, mask_max2);
#endif
        mask = _mm_and_si64(mask, it);
        mask_max = _mm_and_si64(mask_max, it);
        current_min = _mm_or_si64(current_min, mask);
        current_max = _mm_or_si64(current_max, mask_max);
    }

    __m64 left = _m_psllqi(current_min, 16);
    __m64 right = _m_psrlqi(current_min, 48);
    __m64 left2 = _m_psllqi(current_min, 32);
    __m64 right2 = _m_psrlqi(current_min, 32);
    __m64 current2 = _mm_or_si64(left, right);
    __m64 current3 = _mm_or_si64(left2, right2);
    __m64 current4 = _mm_or_si64(_m_psllqi(current_min, 48), _m_psrlqi(current_min, 16));
    __m64 mask = _mm_cmpgt_pi16(current_min, current2);
    __m64 mask2 = _mm_cmpgt_pi16(current_min, current3);
    __m64 mask3 = _mm_cmpgt_pi16(current_min, current4);
    mask = _mm_xor_si64(mask, max_x_4);
#ifdef USE_ANDNOT
    // slower?
    mask = _mm_andnot_si64(mask2, mask);
    mask = _mm_andnot_si64(mask3, mask);
#else
    // faster?
    mask2 = _mm_xor_si64(mask2, max_x_4);
    mask3 = _mm_xor_si64(mask3, max_x_4);

    mask = _mm_and_si64(mask, mask2);
    mask = _mm_and_si64(mask, mask3);
#endif

    current_min = _mm_and_si64(current_min, mask);

    current_min = _mm_or_si64(current_min, _m_psrlqi(current_min, 32));
    current_min = _mm_or_si64(current_min, _m_psrlqi(current_min, 16));

    *out_min = _m_to_int(current_min) & UINT16_MAX;


    left = _m_psllqi(current_min, 16);
    right = _m_psrlqi(current_min, 48);
    left2 = _m_psllqi(current_min, 32);
    right2 = _m_psrlqi(current_min, 32);
    current2 = _mm_or_si64(left, right);
    current3 = _mm_or_si64(left2, right2);
    current4 = _mm_or_si64(_m_psllqi(current_max, 48), _m_psrlqi(current_max, 16));
    mask = _mm_cmpgt_pi16(current_max, current2);
    mask2 = _mm_cmpgt_pi16(current_max, current3);
    mask3 = _mm_cmpgt_pi16(current_max, current4);

    mask = _mm_and_si64(mask, mask2);
    mask = _mm_and_si64(mask, mask3);

    current_max = _mm_and_si64(current_max, mask);

    current_max = _mm_or_si64(current_max, _m_psrlqi(current_max, 32));
    current_max = _mm_or_si64(current_max, _m_psrlqi(current_max, 16));

    *out_max = _m_to_int(current_max) & UINT16_MAX;

    ANY_EMMS();
}


size_t vec_i8v32n_get_min_index(size_t size, int8_t *src)
;
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src)
;
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max)
;


int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 8;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 max_x_8 = _mm_set1_pi8(UINT8_MAX);
    __m64 current = _mm_set1_pi8(INT8_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi8(current, it);

        current = _mm_andnot_si64(mask, current);
        it = _mm_and_si64(it, mask);
        current = _mm_or_si64(current, it);
    }

    __m64 current2 = _mm_or_si64(_m_psllqi(current, 8), _m_psrlqi(current, 56));
    __m64 current3 = _mm_or_si64(_m_psllqi(current, 16), _m_psrlqi(current, 48));
    __m64 current4 = _mm_or_si64(_m_psllqi(current, 24), _m_psrlqi(current, 40));
    __m64 current5 = _mm_or_si64(_m_psllqi(current, 32), _m_psrlqi(current, 32));
    __m64 current6 = _mm_or_si64(_m_psllqi(current, 40), _m_psrlqi(current, 24));
    __m64 current7 = _mm_or_si64(_m_psllqi(current, 48), _m_psrlqi(current, 16));
    __m64 current8 = _mm_or_si64(_m_psllqi(current, 56), _m_psrlqi(current, 8));
    __m64 mask = _mm_cmpgt_pi8(current, current2);
    __m64 mask2 = _mm_cmpgt_pi8(current, current3);
    __m64 mask3 = _mm_cmpgt_pi8(current, current4);
    __m64 mask4 = _mm_cmpgt_pi8(current, current5);
    __m64 mask5 = _mm_cmpgt_pi8(current, current6);
    __m64 mask6 = _mm_cmpgt_pi8(current, current7);
    __m64 mask7 = _mm_cmpgt_pi8(current, current8);
    mask = _mm_xor_si64(mask, max_x_8);
    mask2 = _mm_xor_si64(mask2, max_x_8);
    mask3 = _mm_xor_si64(mask3, max_x_8);

    mask = _mm_andnot_si64(mask4, mask);
    mask2 = _mm_andnot_si64(mask5, mask2);
    mask3 = _mm_andnot_si64(mask6, mask3);
    mask = _mm_andnot_si64(mask7, mask);

    mask2 = _mm_and_si64(mask2, mask3);
    mask = _mm_and_si64(mask, mask2);

    current = _mm_and_si64(current, mask);

    current = _mm_or_si64(current, _m_psrlqi(current, 32));
    current = _mm_or_si64(current, _m_psrlqi(current, 16));
    current = _mm_or_si64(current, _m_psrlqi(current, 8));

    int8_t result = _m_to_int(current) & UINT8_MAX;

    ANY_EMMS();
    return result;
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 8;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 max_x_8 = _mm_set1_pi8(UINT8_MAX);
    __m64 current = _mm_set1_pi8(INT8_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi8(it, current);

        current = _mm_andnot_si64(mask, current);
        it = _mm_and_si64(mask, it);
        current = _mm_or_si64(current, it);
    }

    __m64 current2 = _mm_or_si64(_m_psllqi(current, 8), _m_psrlqi(current, 56));
    __m64 current3 = _mm_or_si64(_m_psllqi(current, 16), _m_psrlqi(current, 48));
    __m64 current4 = _mm_or_si64(_m_psllqi(current, 24), _m_psrlqi(current, 40));
    __m64 current5 = _mm_or_si64(_m_psllqi(current, 32), _m_psrlqi(current, 32));
    __m64 current6 = _mm_or_si64(_m_psllqi(current, 40), _m_psrlqi(current, 24));
    __m64 current7 = _mm_or_si64(_m_psllqi(current, 48), _m_psrlqi(current, 16));
    __m64 current8 = _mm_or_si64(_m_psllqi(current, 56), _m_psrlqi(current, 8));
    __m64 mask = _mm_cmpgt_pi8(current2, current);
    __m64 mask2 = _mm_cmpgt_pi8(current3, current);
    __m64 mask3 = _mm_cmpgt_pi8(current4, current);
    __m64 mask4 = _mm_cmpgt_pi8(current5, current);
    __m64 mask5 = _mm_cmpgt_pi8(current6, current);
    __m64 mask6 = _mm_cmpgt_pi8(current7, current);
    __m64 mask7 = _mm_cmpgt_pi8(current8, current);
    mask = _mm_xor_si64(mask, max_x_8);
    mask2 = _mm_xor_si64(mask2, max_x_8);
    mask3 = _mm_xor_si64(mask3, max_x_8);

    mask = _mm_andnot_si64(mask4, mask);
    mask2 = _mm_andnot_si64(mask5, mask2);
    mask3 = _mm_andnot_si64(mask6, mask3);
    mask = _mm_andnot_si64(mask7, mask);
    mask2 = _mm_and_si64(mask2, mask3);
    mask = _mm_and_si64(mask, mask2);

    current = _mm_and_si64(current, mask);

    current = _mm_or_si64(current, _m_psrlqi(current, 32));
    current = _mm_or_si64(current, _m_psrlqi(current, 16));
    current = _mm_or_si64(current, _m_psrlqi(current, 8));

    int8_t result = _m_to_int(current) & UINT8_MAX;

    ANY_EMMS();
    return result;
}

void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 8;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 max_x_8 = _mm_set1_pi8(UINT8_MAX);
    __m64 current_min = _mm_set1_pi8(INT8_MAX);
    __m64 current_max = _mm_set1_pi8(INT8_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask_min = _mm_cmpgt_pi8(current_min, it);
        __m64 mask_max = _mm_cmpgt_pi8(it, current_max);

        current_min = _mm_andnot_si64(mask_min, current_min);
        current_max = _mm_andnot_si64(mask_max, current_max);
        mask_min = _mm_and_si64(mask_min, it);
        mask_max = _mm_and_si64(mask_max, it);
        current_min = _mm_or_si64(current_min, mask_min);
        current_max = _mm_or_si64(current_max, mask_max);
    }

    __m64 current2 = _mm_or_si64(_m_psllqi(current_min, 8), _m_psrlqi(current_min, 56));
    __m64 current3 = _mm_or_si64(_m_psllqi(current_min, 16), _m_psrlqi(current_min, 48));
    __m64 current4 = _mm_or_si64(_m_psllqi(current_min, 24), _m_psrlqi(current_min, 40));
    __m64 current5 = _mm_or_si64(_m_psllqi(current_min, 32), _m_psrlqi(current_min, 32));
    __m64 current6 = _mm_or_si64(_m_psllqi(current_min, 40), _m_psrlqi(current_min, 24));
    __m64 current7 = _mm_or_si64(_m_psllqi(current_min, 48), _m_psrlqi(current_min, 16));
    __m64 current8 = _mm_or_si64(_m_psllqi(current_min, 56), _m_psrlqi(current_min, 8));
    __m64 mask = _mm_cmpgt_pi8(current_min, current2);
    __m64 mask2 = _mm_cmpgt_pi8(current_min, current3);
    __m64 mask3 = _mm_cmpgt_pi8(current_min, current4);
    __m64 mask4 = _mm_cmpgt_pi8(current_min, current5);
    __m64 mask5 = _mm_cmpgt_pi8(current_min, current6);
    __m64 mask6 = _mm_cmpgt_pi8(current_min, current7);
    __m64 mask7 = _mm_cmpgt_pi8(current_min, current8);
    mask = _mm_xor_si64(mask, max_x_8);
    mask2 = _mm_xor_si64(mask2, max_x_8);
    mask3 = _mm_xor_si64(mask3, max_x_8);

    mask = _mm_andnot_si64(mask4, mask);
    mask2 = _mm_andnot_si64(mask5, mask2);
    mask3 = _mm_andnot_si64(mask6, mask3);
    mask = _mm_andnot_si64(mask7, mask);
    mask2 = _mm_and_si64(mask2, mask3);
    mask = _mm_and_si64(mask, mask2);

    current_min = _mm_and_si64(current_min, mask);

    current_min = _mm_or_si64(current_min, _m_psrlqi(current_min, 32));
    current_min = _mm_or_si64(current_min, _m_psrlqi(current_min, 16));
    current_min = _mm_or_si64(current_min, _m_psrlqi(current_min, 8));

    *out_min = _m_to_int(current_min) & UINT8_MAX;


    current2 = _mm_or_si64(_m_psllqi(current_max, 8), _m_psrlqi(current_max, 56));
    current3 = _mm_or_si64(_m_psllqi(current_max, 16), _m_psrlqi(current_max, 48));
    current4 = _mm_or_si64(_m_psllqi(current_max, 24), _m_psrlqi(current_max, 40));
    current5 = _mm_or_si64(_m_psllqi(current_max, 32), _m_psrlqi(current_max, 32));
    current6 = _mm_or_si64(_m_psllqi(current_max, 40), _m_psrlqi(current_max, 24));
    current7 = _mm_or_si64(_m_psllqi(current_max, 48), _m_psrlqi(current_max, 16));
    current8 = _mm_or_si64(_m_psllqi(current_max, 56), _m_psrlqi(current_max, 8));

    mask = _mm_cmpgt_pi8(current2, current_max);
    mask2 = _mm_cmpgt_pi8(current3, current_max);
    mask3 = _mm_cmpgt_pi8(current4, current_max);
    mask4 = _mm_cmpgt_pi8(current5, current_max);
    mask5 = _mm_cmpgt_pi8(current6, current_max);
    mask6 = _mm_cmpgt_pi8(current7, current_max);
    mask7 = _mm_cmpgt_pi8(current8, current_max);
    mask = _mm_xor_si64(mask, max_x_8);
    mask2 = _mm_xor_si64(mask2, max_x_8);
    mask3 = _mm_xor_si64(mask3, max_x_8);

    mask = _mm_andnot_si64(mask4, mask);
    mask2 = _mm_andnot_si64(mask5, mask2);
    mask3 = _mm_andnot_si64(mask6, mask3);
    mask = _mm_andnot_si64(mask7, mask);
    mask2 = _mm_and_si64(mask2, mask3);
    mask = _mm_and_si64(mask, mask2);

    current_max = _mm_and_si64(current_max, mask);

    current_max = _mm_or_si64(current_max, _m_psrlqi(current_max, 32));
    current_max = _mm_or_si64(current_max, _m_psrlqi(current_max, 16));
    current_max = _mm_or_si64(current_max, _m_psrlqi(current_max, 8));

    *out_max = _m_to_int(current_max) & UINT8_MAX;

    ANY_EMMS();
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
