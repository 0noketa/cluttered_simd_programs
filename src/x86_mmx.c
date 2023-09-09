#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#include "../include/simd_tools.h"

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif

#define USE_ANDNOT

/* local */

static void dump(const char *s, __m64 current)
{
    ANY_EMMS();
 
    fputs(s, stdout);
    int16_t *p = (int16_t*)&current;
    for (int i = 0; i < 4; ++i)
    {
        int it = (int16_t)p[i];
        printf("%d,", it);
    }

    printf("\n");
}
static void dump8(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);
    int it = (int8_t)(_m_to_int(current) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(current) >> 8) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(current) >> 16) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(current) >> 24) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)(_m_to_int(_m_psrlqi(current, 32)) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 8) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 16) & UINT8_MAX);
    printf("%d,", it);
    it = (int8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 24) & UINT8_MAX);
    printf("%d\n", it);
}
static void dump_u8(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);
    unsigned it = (uint8_t)(_m_to_int(current) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(current) >> 8) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(current) >> 16) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(current) >> 24) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)(_m_to_int(_m_psrlqi(current, 32)) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 8) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 16) & UINT8_MAX);
    printf("%u,", it);
    it = (uint8_t)((_m_to_int(_m_psrlqi(current, 32)) >> 24) & UINT8_MAX);
    printf("%u\n", it);
}


/* minmax */

size_t get_min_index(size_t size, int16_t *src)
;

size_t get_max_index(size_t size, int16_t *src)
;


void get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
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


void  vec_i16v16n_abs(size_t size, int16_t *src)
{
    size_t units = size / 4;
	__m64 *p = (__m64*)src;

	ANY_EMMS();
	__m64 max_x_8 = _mm_set1_pi16(UINT16_MAX);
	__m64 minus_one_x_8 = _mm_set1_pi16(-1);
	__m64 zero_x_8 = _mm_setzero_si64();

	for (size_t i = 0; i < units; ++i)
	{
		__m64 it = p[i];

		__m64 mask = _mm_cmpgt_pi16(it, minus_one_x_8);
		__m64 modified = _mm_sub_pi16(zero_x_8, it);
		it = _mm_and_si64(it, mask);
		modified = _mm_andnot_si64(mask, modified);
		p[i] = _mm_or_si64(it, modified);
	}

	ANY_EMMS();
}




void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
{
	size_t units = size / 4;
	__m64 *p = (__m64*)base;
	__m64 *q = (__m64*)target;
	__m64 *r = (__m64*)dst;

	ANY_EMMS();

	for (size_t i = 0; i < units; ++i)
	{
		*r = _mm_subs_pi16(*q, *p);
		
		++p;
		++q;
		++r;
	}

	ANY_EMMS();
}





void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst)
{
	size_t units = size / 4;
	__m64 *p = (__m64*)src1;
	__m64 *q = (__m64*)src2;
	__m64 *r = (__m64*)dst;

	__m64 umax_x_4 = _mm_set1_pi16(UINT16_MAX);
	__m64 min_x_4 = _mm_set1_pi16(INT16_MIN);
	__m64 max_x_4 = _mm_set1_pi16(INT16_MAX);

	ANY_EMMS();

	for (__m64 *p_end = p + units; p < p_end;)
	{
		__m64 it = _mm_subs_pi16(*q, *p);

		__m64 mask_min = _mm_cmpeq_pi16(it, min_x_4);
		__m64 mask_positive = _mm_cmpgt_pi16(it, umax_x_4);

		__m64 zero_x_4 = _mm_setzero_si64();
		__m64 notmin_it = _mm_sub_pi16(zero_x_4, it);
		notmin_it = _mm_andnot_si64(mask_positive, notmin_it);
		it = _mm_and_si64(it, mask_positive);
		notmin_it = _mm_or_si64(notmin_it, it);
		notmin_it = _mm_andnot_si64(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x_8)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm_and_si64(mask_min, max_x_4);  // min->max
		it = _mm_or_si64(it, notmin_it);

		*r = it;

		++p;
		++q;
		++r;
	}

	ANY_EMMS();
}


void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst)
{
	size_t units = size / 2;
	__m64 *p = (__m64*)src;
	__m64 *q = (__m64*)dst;

	ANY_EMMS();
	size_t i = 0;
	size_t j = units - 1;
	
	while (i < j)
	{
		__m64 left = p[i];
		__m64 right = p[j];

#ifdef __3dNOW__
		left = _m_pswapd(left);
		right = _m_pswapd(right);
#else
		__m64 left_left = _mm_slli_si64(left, 32);
		__m64 right_left = _mm_slli_si64(right, 32);
		__m64 left_right = _mm_srli_si64(left, 32);
		__m64 right_right = _mm_srli_si64(right, 32);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);
#endif

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		__m64 it = p[i];

#ifdef __3dNOW__
		it = _m_pswapd(it);
#else
		__m64 left = _mm_slli_si64(it, 32);
		__m64 right = _mm_srli_si64(it, 32);

		it = _mm_or_si64(left, right);
#endif

		q[i] = it;
	}
	
	ANY_EMMS();
}
// current version is slow as generic version is.
void vec_i16v16n_get_reversed(size_t size, int16_t *src, int16_t *dst)
{
	size_t units = size / 4;
	__m64 *p = (__m64*)src;
	__m64 *q = (__m64*)dst;

	ANY_EMMS();
	size_t i = 0;
	size_t j = units - 1;

	while (i < j)
	{
		__m64 left = p[i];
		__m64 right = p[j];

		// every var name means bytes' order
		// mmx has not int64

#ifdef __3dNOW__
		left = _m_pswapd(left);
		right = _m_pswapd(right);
#else
		__m64 left_left = _mm_slli_si64(left, 32);
		__m64 right_left = _mm_slli_si64(right, 32);
		__m64 left_right = _mm_srli_si64(left, 32);
		__m64 right_right = _mm_srli_si64(right, 32);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);
#endif

		left_left = _mm_srli_pi32(left, 16);
		right_left = _mm_srli_pi32(right, 16);
		left_right = _mm_slli_pi32(left, 16);
		right_right = _mm_slli_pi32(right, 16);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		__m64 it = p[i];

#ifdef __3dNOW__
		it = _m_pswapd(it);
#else
		__m64 left = _mm_slli_si64(it, 32);
		__m64 right = _mm_srli_si64(it, 32);

		it = _mm_or_si64(left, right);
#endif

		left = _mm_srli_pi32(it, 16);
		right = _mm_slli_pi32(it, 16);

		it = _mm_or_si64(left, right);

		q[i] = it;
	}
	
	ANY_EMMS();
}
void vec_i8v32n_get_reversed(size_t size, int8_t *src, int8_t *dst)
{
	size_t units = size / 8;
	__m64 *p = (__m64*)src;
	__m64 *q = (__m64*)dst;

	ANY_EMMS();
	size_t i = 0;
	size_t j = units - 1;

	while (i < j)
	{
		__m64 left = p[i];
		__m64 right = p[j];

		// every var name means bytes' order
		// mmx has not int64

#ifdef __3dNOW__
		left = _m_pswapd(left);
		right = _m_pswapd(right);
#else
		__m64 left_left = _mm_slli_si64(left, 32);
		__m64 right_left = _mm_slli_si64(right, 32);
		__m64 left_right = _mm_srli_si64(left, 32);
		__m64 right_right = _mm_srli_si64(right, 32);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);
#endif

		left_left = _mm_srli_pi32(left, 16);
		right_left = _mm_srli_pi32(right, 16);
		left_right = _mm_slli_pi32(left, 16);
		right_right = _mm_slli_pi32(right, 16);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);

		left_left = _mm_srli_pi16(left, 8);
		right_left = _mm_srli_pi16(right, 8);
		left_right = _mm_slli_pi16(left, 8);
		right_right = _mm_slli_pi16(right, 8);

		left = _mm_or_si64(left_left, left_right);
		right = _mm_or_si64(right_left, right_right);

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		__m64 it = p[i];

#ifdef __3dNOW__
		it = _m_pswapd(it);
#else
		__m64 left = _mm_slli_si64(it, 32);
		__m64 right = _mm_srli_si64(it, 32);

		it = _mm_or_si64(left, right);
#endif

		left = _mm_srli_pi32(it, 16);
		right = _mm_slli_pi32(it, 16);

		it = _mm_or_si64(left, right);

		left = _mm_srli_pi16(it, 8);
		right = _mm_slli_pi16(it, 8);

		it = _mm_or_si64(left, right);

		q[i] = it;
	}
	
	ANY_EMMS();
}




/* shift */

void vec_u256n_shl1(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;

    __m64 mask_high = _mm_set1_pi8(UINT8_MAX - 1);
    __m64 mask_low = _mm_set1_pi8(1);
    __m64 mask_carry = _mm_set_pi8(1,0,0,0, 0,0,0,0);
    __m64 it = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m64 next = p[i + 1];
        __m64 shifted = _mm_slli_si64(it, 1);
        __m64 shifted2 = _mm_srli_si64(it, 7);
        shifted = _mm_and_si64(shifted, mask_high);
        shifted2 = _mm_and_si64(shifted2, mask_low);
        __m64 shifted3 = _mm_slli_si64(next, 64 - 8 - 7);
        shifted2 = _mm_srli_si64(shifted2, 8);

        shifted3 = _mm_and_si64(shifted3, mask_carry);
        __m64 it2 = _mm_or_si64(shifted, shifted2);
        it2 = _mm_or_si64(it2, shifted3);

        q[i] = it2;;

        it = next;
    }

    {
        __m64 shifted = _mm_slli_si64(it, 1);
        __m64 shifted2 = _mm_srli_si64(it, 7);
        shifted = _mm_and_si64(shifted, mask_high);
        shifted2 = _mm_and_si64(shifted2, mask_low);
        shifted2 = _mm_srli_si64(shifted2, 8);

        q[units - 1] = _mm_or_si64(shifted, shifted2);
    }

    ANY_EMMS();
}
void vec_u256n_shl8(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;

    __m64 it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m64 it = p[i + 1];
        __m64 carried = _mm_slli_si64(it, 56);
        __m64 shifted = _mm_srli_si64(it0, 8);
        shifted = _mm_or_si64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = _mm_srli_si64(it0, 8);

    ANY_EMMS();
}
void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;

    __m64 it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        __m64 it = p[i - 1];
        __m64 carried = _mm_srli_si64(it, 56);
        __m64 shifted = _mm_slli_si64(it0, 8);
        shifted = _mm_or_si64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = _mm_slli_si64(it0, 8);

    ANY_EMMS();
}
void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;

    __m64 it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m64 it = p[i + 1];
        __m64 carried = _mm_slli_si64(it, 32);
        __m64 shifted = _mm_srli_si64(it0, 32);
        shifted = _mm_or_si64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = _mm_srli_si64(it0, 32);

    ANY_EMMS();    
}
void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;

    __m64 it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        __m64 it = p[i + 1];
        __m64 carried = _mm_srli_si64(it, 32);
        __m64 shifted = _mm_slli_si64(it0, 32);
        shifted = _mm_or_si64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = _mm_srli_si64(it0, 32);

    ANY_EMMS();    
}
void vec_u256n_rol1(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_rol8(size, src, dst);

    dst[size - 1] = src[0] >> 7;
}
void vec_u256n_rol8(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shl8(size, src, dst);

    dst[size - 1] = src[0];
}
void vec_u256n_ror8(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shr8(size, src, dst);

    dst[0] = src[size - 1];
}

void vec_u256n_rol32(size_t size, uint8_t *src, uint8_t *dst)
;

void vec_u256n_ror32(size_t size, uint8_t *src, uint8_t *dst)
;


/* humming weight */

size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src)
{
    size_t units = size / 8;

    __m64 *p = (__m64*)src;

    __m64 rs = _mm_setzero_si64();

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i++];
        __m64 it2 = p[i];
        __m64 rs0 = _mm_setzero_si64();

        __m64 one_x8 = _mm_set1_pi8(1);

        for (int j = 0; j < 8; ++j)
        {
            __m64 tmp = _mm_and_si64(it, one_x8);
            __m64 tmp2 = _mm_and_si64(it2, one_x8);

            rs0 = _mm_adds_pi8(rs0, tmp);
            rs0 = _mm_adds_pi8(rs0, tmp2);

            it = _mm_srli_si64(it, 1);
            it2 = _mm_srli_si64(it2, 1);
        }

        __m64 mask = _mm_set1_pi32(0x000000FF);
        __m64 mask2 = _mm_set1_pi32(0x0000FF00);
        __m64 mask3 = _mm_set1_pi32(0x00FF0000);
        __m64 mask4 = _mm_set1_pi32(0xFF000000);

        __m64 masked = _mm_and_si64(mask, rs0);
        __m64 masked2 = _mm_and_si64(mask2, rs0);
        __m64 masked3 = _mm_and_si64(mask3, rs0);
        __m64 masked4 = _mm_and_si64(mask4, rs0);
        masked2 = _mm_srli_si64(masked2, 8);
        masked3 = _mm_srli_si64(masked3, 16);
        masked4 = _mm_srli_si64(masked4, 24);

        masked = _mm_add_pi32(masked, masked3);
        masked2 = _mm_add_pi32(masked2, masked4);

        rs = _mm_add_pi32(rs, masked);
        rs = _mm_add_pi32(rs, masked2);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}


/* sorted arrays */

void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 4;
	__m64 *p = (__m64*)src;

	ANY_EMMS();
	__m64 one_x_4 = _mm_set1_pi16(1);
	__m64 element_x_4 = _mm_set1_pi16(element);
	__m64 start0 = _mm_setzero_si64();
	__m64 end0 = _mm_setzero_si64();

	for (int i = 0; i < units; ++i)
	{
		__m64 it = p[i];
		__m64 mask_under = _mm_cmpgt_pi16(element_x_4, it);
		__m64 mask_over = _mm_cmpgt_pi16(it, element_x_4);
		__m64 masked_under = _mm_and_si64(mask_under, one_x_4);
		__m64 masked_over = _mm_and_si64(mask_over, one_x_4);
		start0 = _mm_adds_pi16(start0, masked_under);
		end0 = _mm_adds_pi16(end0, masked_over);
	}
		
	__m64 start1 = _mm_srli_si64(start0, 32);
	__m64 end1 = _mm_srli_si64(end0, 32);
	start0 = _mm_adds_pi16(start0, start1);
	end0 = _mm_adds_pi16(end0, end1);
	start1 = _mm_srli_si64(start0, 16);
	end1 = _mm_srli_si64(end0, 16);
	start0 = _mm_adds_pi16(start0, start1);
	end0 = _mm_adds_pi16(end0, end1);

	*out_start = _mm_cvtsi64_si32(start0) & UINT16_MAX;
	int16_t end2 = _mm_cvtsi64_si32(end0) & UINT16_MAX;
	*out_end = size - end2;

	ANY_EMMS();
}


int  vec_i16v16n_is_sorted_a(size_t size, int16_t *src)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 it0 = _mm_set1_pi16(INT16_MIN);
    __m64 cond0 = _mm_set1_pi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 it_left = _m_psllqi(it, 16);
        it0 = _m_psrlqi(it0, 48);
        it_left = _mm_or_si64(it0, it_left);

        __m64 cond = _mm_cmpgt_pi16(it_left, it);
        
        cond0 = _mm_andnot_si64(cond, cond0);
        it0 = it;
    }

    __m64 cond_right = _mm_srli_si64(cond0, 32);
    cond0 = _mm_and_si64(cond0, cond_right);
    cond_right = _mm_srli_si64(cond0, 16);
    cond0 = _mm_and_si64(cond0, cond_right);

    int result = _m_to_int(cond0) & UINT16_MAX;

    ANY_EMMS();
    return !!result;
}
int  vec_i16v16n_is_sorted_d(size_t size, int16_t *src)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 it0 = _mm_set1_pi16(INT16_MAX);
    __m64 cond0 = _mm_set1_pi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 it_left = _m_psllqi(it, 16);
        it0 = _m_psrlqi(it0, 48);
        it_left = _mm_or_si64(it0, it_left);

        __m64 cond = _mm_cmpgt_pi16(it, it_left);
        
        cond0 = _mm_andnot_si64(cond, cond0);
        it0 = it;
    }

    __m64 cond_right = _mm_srli_si64(cond0, 32);
    cond0 = _mm_and_si64(cond0, cond_right);
    cond_right = _mm_srli_si64(cond0, 16);
    cond0 = _mm_and_si64(cond0, cond_right);

    int result = _m_to_int(cond0) & UINT16_MAX;

    ANY_EMMS();
    return !!result;
}
int  vec_i16v16n_is_sorted(size_t size, int16_t *src)
{
    size_t units = size / 4;
    __m64 *p = (__m64*)src;

    ANY_EMMS();
    __m64 it_a0 = _mm_set1_pi16(INT16_MIN);
    __m64 it_d0 = _mm_set1_pi16(INT16_MAX);
    __m64 cond_a0 = _mm_set1_pi16(-1);
    __m64 cond_d0 = _mm_set1_pi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 it_left = _m_psllqi(it, 16);
        it_a0 = _m_psrlqi(it_a0, 48);
        it_d0 = _m_psrlqi(it_d0, 48);
        __m64 it_left_a = _mm_or_si64(it_a0, it_left);
        __m64 it_left_d = _mm_or_si64(it_d0, it_left);

        __m64 cond_a = _mm_cmpgt_pi16(it_left_a, it);
        __m64 cond_d = _mm_cmpgt_pi16(it, it_left_d);
        
        cond_a0 = _mm_andnot_si64(cond_a, cond_a0);
        cond_d0 = _mm_andnot_si64(cond_d, cond_d0);
        it_a0 = it;
        it_d0 = it;
    }

    __m64 cond_a_right = _mm_srli_si64(cond_a0, 32);
    __m64 cond_d_right = _mm_srli_si64(cond_d0, 32);
    cond_a0 = _mm_and_si64(cond_a0, cond_a_right);
    cond_d0 = _mm_and_si64(cond_d0, cond_d_right);
    cond_a_right = _mm_srli_si64(cond_a0, 16);
    cond_d_right = _mm_srli_si64(cond_d0, 16);
    cond_a0 = _mm_and_si64(cond_a0, cond_a_right);
    cond_d0 = _mm_and_si64(cond_d0, cond_d_right);

    int result_a = _m_to_int(cond_a0) & UINT16_MAX;
    int result_d = _m_to_int(cond_d0) & UINT16_MAX;

    ANY_EMMS();
    return result_a || result_d;
}


// data type: [a0, b0, ..., a1, b1, ...]
void  vec_i16v4x2n_bubblesort(size_t size, int16_t *src, int16_t *dst)
;
/*
{
	size_t units = size / 4;
	__m64 *p = (__m64*)src;
	__m64 *q = (__m64*)dst;

	int modified = 0;
	do
	{
		__m64 modified0 = _mm_setzero_si64();

		__m64 it = p[0];
		for (size_t i = 0; i < units - 1; ++i)
		{
			__m64 nxt = p[i + 1];
			
			__m64 it2 = _mm_min_epi16(it, nxt);
			__m64 nxt2 = _mm_max_epi16(it, nxt);
			
			__m64 modified1 = _mm_xor_si64(it, it2);
			__m64 modified2 = _mm_xor_si64(nxt, nxt2);
			modified0 = _mm_or_si64(modified0, modified1);
			modified0 = _mm_or_si64(modified0, modified2);

			q[i] = it2;
			it = nxt2;
		}
		q[units - 1] = it;
		p = q;

		__m64 modified1 = _mm_srli_si64(modified0, 8);
		modified0 = _mm_or_si64(modified0, modified1);
		modified1 = _mm_srli_si64(modified0, 4);
		modified0 = _mm_or_si64(modified0, modified1);
		
		modified = _m_to_int(modified0);
	}
	while (modified);
}
*/
