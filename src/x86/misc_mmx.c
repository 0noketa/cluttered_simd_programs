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

#ifndef USE_ANDNOT
#define USE_ANDNOT
#endif

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






/* assignment */

void vec_i32v8n_set_seq(size_t size, int32_t *src, int32_t start, int32_t diff)
;

