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


#include <stdio.h>
static void dump_u8(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);
    uint8_t *p = (void*)&current;
    int it;
    for (int i = 0; i < 8; ++i)
    {
        it = p[i];
        printf("%c(%02x),", isprint(it) ? it : '.', it);
    }
    puts("");
}


/* min/max */

size_t vec_i32x8n_get_min_index(size_t size, const int32_t *src);
size_t vec_i32x8n_get_max_index(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32x8n_get_min(size_t size, const int32_t *src)
;
int32_t vec_i32x8n_get_max(size_t size, const int32_t *src);
void vec_i32x8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max);


static void vec_i16x16n_get_min_and_index_i16_i(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_idx)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

    __m64 iota = _mm_set_pi16(3, 2, 1, 0);
    __m64 four_x4 = _mm_set1_pi16(4);
    __m64 max_x4 = _mm_set1_pi16(UINT16_MAX);
    __m64 current = _mm_set1_pi16(INT16_MAX);
    __m64 current_idx = _mm_set1_pi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(current, it);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
        current_idx = _mm_andnot_si64(mask, current_idx);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        current = _mm_and_si64(current, mask2);
        current_idx = _mm_and_si64(current_idx, mask2);
#endif
        it = _mm_and_si64(it, mask);
        __m64 idx = _mm_and_si64(iota, mask);
        current = _mm_or_si64(current, it);
        current_idx = _mm_or_si64(current_idx, idx);

        iota = _mm_adds_pi16(iota, four_x4);
    }

    {
        __m64 it = _mm_srli_si64(current, 32);
        __m64 idx = _mm_srli_si64(current_idx, 32);
        __m64 mask = _mm_cmpgt_pi16(current, it);
    #ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
        current_idx = _mm_andnot_si64(mask, current_idx);
    #else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        current = _mm_and_si64(current, mask2);
        current_idx = _mm_and_si64(current_idx, mask2);
    #endif
        it = _mm_and_si64(it, mask);
        idx = _mm_and_si64(idx, mask);
        current = _mm_or_si64(current, it);
        current_idx = _mm_or_si64(current_idx, idx);
    }

    uint32_t results = _mm_cvtsi64_si32(current);
    uint32_t results_idx = _mm_cvtsi64_si32(current_idx);
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
    ANY_EMMS();
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

    ANY_EMMS();
    return current_min_index;
}

static void vec_i16x16n_get_max_and_index_i16_i(size_t size, const int16_t *src, int16_t *out_max, int16_t *out_idx)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

    __m64 iota = _mm_set_pi16(3, 2, 1, 0);
    __m64 four_x4 = _mm_set1_pi16(4);
    __m64 max_x4 = _mm_set1_pi16(UINT16_MAX);
    __m64 current = _mm_set1_pi16(INT16_MIN);
    __m64 current_idx = _mm_set1_pi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(it, current);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
        current_idx = _mm_andnot_si64(mask, current_idx);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        current = _mm_and_si64(current, mask2);
        current_idx = _mm_and_si64(current_idx, mask2);
#endif
        it = _mm_and_si64(mask, it);
        __m64 idx = _mm_and_si64(mask, iota);
        current = _mm_or_si64(current, it);
        current_idx = _mm_or_si64(current_idx, idx);

        iota = _mm_adds_pi16(iota, four_x4);
    }

    {
        __m64 it = _mm_srli_si64(current, 32);
        __m64 idx = _mm_srli_si64(current_idx, 32);
        __m64 mask = _mm_cmpgt_pi16(it, current);
    #ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
        current_idx = _mm_andnot_si64(mask, current_idx);
    #else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        current = _mm_and_si64(current, mask2);
        current_idx = _mm_and_si64(current_idx, mask2);
    #endif
        it = _mm_and_si64(it, mask);
        idx = _mm_and_si64(idx, mask);
        current = _mm_or_si64(current, it);
        current_idx = _mm_or_si64(current_idx, idx);
    }

    uint32_t results = _mm_cvtsi64_si32(current);
    uint32_t results_idx = _mm_cvtsi64_si32(current_idx);
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
    ANY_EMMS();
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

    ANY_EMMS();
    return current_max_index;
}
void vec_i16x16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max)
;

int16_t vec_i16x16n_get_min(size_t size, const int16_t *src)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

    // ANY_EMMS();
    __m64 max_x4 = _mm_set1_pi16(UINT16_MAX);
    __m64 current = _mm_set1_pi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(current, it);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
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
    mask = _mm_xor_si64(mask, max_x4);
#ifdef USE_ANDNOT
    mask = _mm_andnot_si64(mask2, mask);
    mask = _mm_andnot_si64(mask3, mask);
#else
    mask2 = _mm_xor_si64(mask2, max_x4);
    mask3 = _mm_xor_si64(mask3, max_x4);

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

int16_t vec_i16x16n_get_max(size_t size, const int16_t *src)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

    ANY_EMMS();
#ifdef USE_ANDNOT
    __m64 max_x4 = _mm_set1_pi16(UINT16_MAX);
#endif
    __m64 current = _mm_set1_pi16(INT16_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi16(it, current);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        current = _mm_and_si64(current, mask2);
#endif
        it = _mm_and_si64(mask, it);
        current = _mm_or_si64(current, it);
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

void vec_i16x16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

    __m64 max_x4 = _mm_set1_pi16(UINT16_MAX);
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
        __m64 mask2 = _mm_xor_si64(mask, max_x4);
        __m64 mask_max2 = _mm_xor_si64(mask_max, max_x4);

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
    mask = _mm_xor_si64(mask, max_x4);
#ifdef USE_ANDNOT
    // slower?
    mask = _mm_andnot_si64(mask2, mask);
    mask = _mm_andnot_si64(mask3, mask);
#else
    // faster?
    mask2 = _mm_xor_si64(mask2, max_x4);
    mask3 = _mm_xor_si64(mask3, max_x4);

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


int8_t vec_i8x32n_get_min_index_i8(size_t size, const int8_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const __m64*)src;

    __m64 iota = _mm_set_pi8(7, 6, 5, 4,  3, 2, 1, 0);
    __m64 eight_x8 = _mm_set1_pi8(8);
    __m64 max_x8 = _mm_set1_pi8(UINT8_MAX);
    __m64 current = _mm_set1_pi8(INT8_MAX);
    __m64 current_idx = _mm_set1_pi8(INT8_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 mask = _mm_cmpgt_pi8(current, it);
#ifdef USE_ANDNOT
        current = _mm_andnot_si64(mask, current);
        current_idx = _mm_andnot_si64(mask, current_idx);
#else
        __m64 mask2 = _mm_xor_si64(mask, max_x8);
        current = _mm_and_si64(current, mask2);
        current_idx = _mm_and_si64(current_idx, mask2);
#endif
        it = _mm_and_si64(it, mask);
        __m64 it2 = _mm_and_si64(iota, mask);
        current = _mm_or_si64(current, it);
        current_idx = _mm_or_si64(current_idx, it2);

        iota = _mm_add_pi8(iota, eight_x8);
    }

    __m64 current2 = _mm_srli_si64(current, 32);
    __m64 current2_idx = _mm_srli_si64(current_idx, 32);
    __m64 mask = _mm_cmpgt_pi8(current, current2);

    current2 = _mm_and_si64(mask, current2);
    current2_idx = _mm_and_si64(mask, current2_idx);
#ifdef USE_ANDNOT
    current = _mm_andnot_si64(mask, current);
    current_idx = _mm_andnot_si64(mask, current_idx);
#else
    mask = _mm_xor_si64(mask, max_x8);

    current = _mm_and_si64(mask, current);
    current_idx = _mm_and_si64(mask, current_idx);
#endif

    current = _mm_or_si64(current, current2);
    current_idx = _mm_or_si64(current_idx, current2_idx);


    current2 = _mm_srli_si64(current, 16);
    current2_idx = _mm_srli_si64(current_idx, 16);
    mask = _mm_cmpgt_pi8(current, current2);

    current2 = _mm_and_si64(mask, current2);
    current2_idx = _mm_and_si64(mask, current2_idx);
#ifdef USE_ANDNOT
    current = _mm_andnot_si64(mask, current);
    current_idx = _mm_andnot_si64(mask, current_idx);
#else
    mask = _mm_xor_si64(mask, max_x8);

    current = _mm_and_si64(mask, current);
    current_idx = _mm_and_si64(mask, current_idx);
#endif

    current = _mm_or_si64(current, current2);
    current_idx = _mm_or_si64(current_idx, current2_idx);

    uint32_t results = _mm_cvtsi64_si32(current);
    uint32_t results_idx = _mm_cvtsi64_si32(current_idx);
    int8_t lo = results & 0xFF;
    int8_t lo_idx = results_idx & 0xFF;
    int8_t hi = (results >> 8) & 0xFF;
    int8_t hi_idx = (results_idx >> 8) & 0xFF;

    ANY_EMMS();
    return lo < lo_idx ? lo_idx : hi_idx;
}
size_t vec_i8x32n_get_min_index(size_t size, const int8_t *src)
;
int8_t vec_i8x32n_get_max_index_i8(size_t size, const int8_t *src)
;
size_t vec_i8x32n_get_max_index(size_t size, const int8_t *src)
;
void vec_i8x32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max)
;


int8_t vec_i8x32n_get_min(size_t size, const int8_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const __m64*)src;

    ANY_EMMS();
    __m64 max_x8 = _mm_set1_pi8(UINT8_MAX);
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
    mask = _mm_xor_si64(mask, max_x8);
    mask2 = _mm_xor_si64(mask2, max_x8);
    mask3 = _mm_xor_si64(mask3, max_x8);

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
int8_t vec_i8x32n_get_max(size_t size, const int8_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const __m64*)src;

    ANY_EMMS();
    __m64 max_x8 = _mm_set1_pi8(UINT8_MAX);
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
    mask = _mm_xor_si64(mask, max_x8);
    mask2 = _mm_xor_si64(mask2, max_x8);
    mask3 = _mm_xor_si64(mask3, max_x8);

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

void vec_i8x32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 8;
    const __m64 *p = (const __m64*)src;

    ANY_EMMS();
    __m64 max_x8 = _mm_set1_pi8(UINT8_MAX);
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
    mask = _mm_xor_si64(mask, max_x8);
    mask2 = _mm_xor_si64(mask2, max_x8);
    mask3 = _mm_xor_si64(mask3, max_x8);

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
    mask = _mm_xor_si64(mask, max_x8);
    mask2 = _mm_xor_si64(mask2, max_x8);
    mask3 = _mm_xor_si64(mask3, max_x8);

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

// no-saturation
static int32_t vec_i32x8n_count_i32_(size_t size, const int32_t *src, int32_t value)
{
    size_t units = size / 2;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 one_x2 = _mm_set1_pi32(1);
    __m64 needle = _mm_set1_pi32(value);

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 mask = _mm_cmpeq_pi32(it, needle);
        __m64 results0 = _mm_and_si64(mask, one_x2);
        results = _mm_add_pi32(results, results0);
    }

    results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
    int32_t result = _mm_cvtsi64_si32(results);
    return result;
}
int32_t vec_i32x8n_count_i32(size_t size, const int32_t *src, int32_t value)
{
    int32_t result = vec_i32x8n_count_i32_(size,src, value);

    ANY_EMMS();
    return result;
}
size_t vec_i32x8n_count(size_t size, const int32_t *src, int32_t value)
{
    const size_t unit_size = 0x20000000;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        size_t result0 = result;
        size_t result2 = vec_i32x8n_count_i32(unit_size, src + i * unit_size, value);
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        size_t result0 = result;
        // includes emms
        size_t result2 = vec_i32x8n_count_i32(size2, src + base, value);
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;

        return result;
    }
    else
    {
        ANY_EMMS();
        return result;
    }
}
// returns u16x4
static __m64 vec_i16x16n_count_m64(size_t size, const int16_t *src, int16_t value)
{
    size_t units = size / 4;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 one_x4 = _mm_set1_pi16(1);
    __m64 needle = _mm_set1_pi16(value);

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 mask = _mm_cmpeq_pi16(it, needle);
        __m64 d = _mm_and_si64(mask, one_x4);
        results = _mm_adds_pu16(results, d);
    }

    return results;
}
int16_t vec_i16x16n_count_i16(size_t size, const int16_t *src, int16_t value)
{
    __m64 results = vec_i16x16n_count_m64(size, src, value);
    results = _mm_adds_pu16(results, _mm_srli_si64(results, 16));
    results = _mm_adds_pu16(results, _mm_srli_si64(results, 32));
    uint16_t result = _mm_cvtsi64_si32(results) & 0xFFFF;

    ANY_EMMS();
    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_i16x16n_count(size_t size, const int16_t *src, int16_t value)
{
    const size_t unit_size = 0x8000 * 4;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m64 results = vec_i16x16n_count_m64(unit_size, src + i * unit_size, value);
        const __m64 mask_lower = _mm_set1_pi32(0x0000FFFF);
        __m64 results2 = _mm_and_si64(_mm_srli_si64(results, 16), mask_lower);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi32(results, results2);
        results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results);

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m64 results = vec_i16x16n_count_m64(size2, src + base, value);
        const __m64 mask_lower = _mm_set1_pi32(0x0000FFFF);
        __m64 results2 = _mm_and_si64(_mm_srli_si64(results, 16), mask_lower);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi32(results, results2);
        results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results);

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    ANY_EMMS();
    return result;
}
// returns u8x8
static __m64 vec_i8x32n_count_m64(size_t size, const int8_t *src, int8_t value)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 one_x8 = _mm_set1_pi8(1);
    __m64 needle = _mm_set1_pi8(value);

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 mask = _mm_cmpeq_pi8(it, needle);
        __m64 results0 = _mm_and_si64(mask, one_x8);
        results = _mm_adds_pu8(results, results0);
    }

    return results;
}
int8_t vec_i8x32n_count_i8(size_t size, const int8_t *src, int8_t value)
{
    __m64 results = vec_i8x32n_count_m64(size, src, value);
    results = _mm_adds_pu8(results, _mm_srli_si64(results, 8));
    results = _mm_adds_pu8(results, _mm_srli_si64(results, 16));
    results = _mm_adds_pu8(results, _mm_srli_si64(results, 32));
    size_t result = _mm_cvtsi64_si32(results) & 0xFFFF;

    ANY_EMMS();
    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_i8x32n_count(size_t size, const int8_t *src, int8_t value)
{
    const size_t unit_size = 0x80 * 8;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m64 results = vec_i8x32n_count_m64(unit_size, src + i * unit_size, value);
        const __m64 mask_lower = _mm_set1_pi16(0x00FF);
        __m64 results2 = _mm_and_si64(_mm_srli_si64(results, 8), mask_lower);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi16(results, results2);
        results = _mm_add_pi16(results, _mm_srli_si64(results, 16));
        results = _mm_add_pi16(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m64 results = vec_i8x32n_count_m64(size2, src + base, value);
        const __m64 mask_lower = _mm_set1_pi16(0x00FF);
        __m64 results2 = _mm_and_si64(_mm_srli_si64(results, 8), mask_lower);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi16(results, results2);
        results = _mm_add_pi16(results, _mm_srli_si64(results, 16));
        results = _mm_add_pi16(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    ANY_EMMS();
    return result;
}
