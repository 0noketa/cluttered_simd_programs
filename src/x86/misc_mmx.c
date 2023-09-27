#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#include "../../include/simd_tools.h"

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif

#ifndef USE_ANDNOT
#define USE_ANDNOT
#endif

/* local */

#if 0
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
    uint8_t *p = (void*)&current;
    int it;
    for (int i = 0; i < 8; ++i)
    {
        it = p[i];
        printf("%c(%02x),", isprint(it) ? it : '.', it);
    }
    puts("");
}
#endif



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







/* hamming weight */

#ifdef USE_128BIT_UNITS
static inline __m64 vec_u256n_get_hamming_weight_i(__m64 it, __m64 it2, __m64 rs,  __m64 mask, __m64 mask2, __m64 mask3, __m64 mask4, __m64 mask5)
{
    __m64 tmp = _mm_srli_si64(it, 1);
    __m64 tmp2 = _mm_srli_si64(it2, 1);
    it = _mm_and_si64(it, mask);
    it2 = _mm_and_si64(it2, mask);
    tmp = _mm_and_si64(tmp, mask);
    tmp2 = _mm_and_si64(tmp2, mask);
    it = _mm_adds_pu16(it, tmp);
    it2 = _mm_adds_pu16(it2, tmp2);

    tmp = _mm_srli_si64(it, 2);
    tmp2 = _mm_srli_si64(it2, 2);
    it = _mm_and_si64(it, mask2);
    it2 = _mm_and_si64(it2, mask2);
    tmp = _mm_and_si64(tmp, mask2);
    tmp2 = _mm_and_si64(tmp2, mask2);
    it = _mm_adds_pu16(it, tmp);
    it2 = _mm_adds_pu16(it2, tmp2);

    tmp = _mm_srli_si64(it, 4);
    tmp2 = _mm_srli_si64(it2, 4);
    it = _mm_and_si64(it, mask3);
    it2 = _mm_and_si64(it2, mask3);
    tmp = _mm_and_si64(tmp, mask3);
    tmp2 = _mm_and_si64(tmp2, mask3);
    it = _mm_adds_pu16(it, tmp);
    it2 = _mm_adds_pu16(it2, tmp2);

    tmp = _mm_srli_si64(it, 8);
    tmp2 = _mm_srli_si64(it2, 8);
    it = _mm_and_si64(it, mask4);
    it2 = _mm_and_si64(it2, mask4);
    tmp = _mm_and_si64(tmp, mask4);
    tmp2 = _mm_and_si64(tmp2, mask4);
    it = _mm_adds_pu16(it, tmp);
    it2 = _mm_adds_pu16(it2, tmp2);

    tmp = _mm_srli_si64(it, 16);
    tmp2 = _mm_srli_si64(it2, 16);
    it = _mm_and_si64(it, mask5);
    it2 = _mm_and_si64(it2, mask5);
    tmp = _mm_and_si64(tmp, mask5);
    tmp2 = _mm_and_si64(tmp2, mask5);
    it = _mm_add_pi32(it, tmp);
    it2 = _mm_add_pi32(it2, tmp2);

    rs = _mm_add_pi32(rs, it);
    rs = _mm_add_pi32(rs, it2);

    return rs;
}
#else
static inline __m64 vec_u256n_get_hamming_weight_i(__m64 it, __m64 rs,  __m64 mask, __m64 mask2, __m64 mask3, __m64 mask4, __m64 mask5)
{
    __m64 tmp = _mm_srli_si64(it, 1);
    it = _mm_and_si64(it, mask);
    tmp = _mm_and_si64(tmp, mask);
    it = _mm_adds_pu16(it, tmp);

    tmp = _mm_srli_si64(it, 2);
    it = _mm_and_si64(it, mask2);
    tmp = _mm_and_si64(tmp, mask2);
    it = _mm_adds_pu16(it, tmp);

    tmp = _mm_srli_si64(it, 4);
    it = _mm_and_si64(it, mask3);
    tmp = _mm_and_si64(tmp, mask3);
    it = _mm_adds_pu16(it, tmp);

    tmp = _mm_srli_si64(it, 8);
    it = _mm_and_si64(it, mask4);
    tmp = _mm_and_si64(tmp, mask4);
    it = _mm_adds_pu16(it, tmp);

    tmp = _mm_srli_si64(it, 16);
    it = _mm_and_si64(it, mask5);
    tmp = _mm_and_si64(tmp, mask5);
    it = _mm_add_pi32(it, tmp);

    rs = _mm_add_pi32(rs, it);

    return rs;
}
#endif
size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src)
#ifdef USE_128BIT_UNITS
{
    size_t units = size / 8 / 2;

    __m64 *p = (__m64*)src;

    __m64 rs = _mm_setzero_si64();

    __m64 mask = _mm_set1_pi32(0x55555555);
    __m64 mask2 = _mm_set1_pi32(0x33333333);
    __m64 mask3 = _mm_set1_pi32(0x0F0F0F0F);
    __m64 mask4 = _mm_set1_pi32(0x00FF00FF);
    __m64 mask5 = _mm_set1_pi32(0x0000FFFF);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}
#else
{
    size_t units = size / 8;

    __m64 *p = (__m64*)src;

    __m64 rs = _mm_setzero_si64();

    __m64 mask = _mm_set1_pi32(0x55555555);
    __m64 mask2 = _mm_set1_pi32(0x33333333);
    __m64 mask3 = _mm_set1_pi32(0x0F0F0F0F);
    __m64 mask4 = _mm_set1_pi32(0x00FF00FF);
    __m64 mask5 = _mm_set1_pi32(0x0000FFFF);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        rs = vec_u256n_get_hamming_weight_i(it, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}
#endif

size_t vec_u256n_get_hamming_distance(size_t size, uint8_t *src1, uint8_t *src2)
#ifdef USE_128BIT_UNITS
{
    size_t units = size / 8 / 2;

    __m64 *p = (__m64*)src1;
    __m64 *q = (__m64*)src2;

    __m64 rs = _mm_setzero_si64();

    __m64 mask = _mm_set1_pi32(0x55555555);
    __m64 mask2 = _mm_set1_pi32(0x33333333);
    __m64 mask3 = _mm_set1_pi32(0x0F0F0F0F);
    __m64 mask4 = _mm_set1_pi32(0x00FF00FF);
    __m64 mask5 = _mm_set1_pi32(0x0000FFFF);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];
        __m64 it_b = q[i * 2];
        __m64 it2_b = q[i * 2 + 1];

        it = _mm_xor_si64(it, it_b);
        it2 = _mm_xor_si64(it2, it2_b);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}
#else
{
    size_t units = size / 8;

    __m64 *p = (__m64*)src1;
    __m64 *q = (__m64*)src2;

    __m64 rs = _mm_setzero_si64();

    __m64 mask = _mm_set1_pi32(0x55555555);
    __m64 mask2 = _mm_set1_pi32(0x33333333);
    __m64 mask3 = _mm_set1_pi32(0x0F0F0F0F);
    __m64 mask4 = _mm_set1_pi32(0x00FF00FF);
    __m64 mask5 = _mm_set1_pi32(0x0000FFFF);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 it_b = q[i];

        it = _mm_xor_si64(it, it_b);

        rs = vec_u256n_get_hamming_weight_i(it, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}
#endif


size_t vec_i32v8n_get_sum(size_t size, uint32_t *src)
{
	size_t units = size / 2 / 4;
	__m64 *p = (__m64*)src;

    __m64 rs = _mm_setzero_si64();

	for (size_t i = 0; i < units; ++i)
	{
        __m64 it0 = p[i * 4];
        __m64 it1 = p[i * 4 + 1];
        __m64 it2 = p[i * 4 + 2];
        __m64 it3 = p[i * 4 + 3];

        it0 = _mm_add_pi32(it0, it1);
        it2 = _mm_add_pi32(it2, it3);
        it0 = _mm_add_pi32(it0, it2);
        rs = _mm_add_pi32(rs, it0);
	}

    size_t r = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    r += _mm_cvtsi64_si32(rs);

	ANY_EMMS();
    return r;
}
int16_t vec_i16v16n_get_sum_i16(size_t size, uint16_t *src)
;
size_t vec_i16v16n_get_sum(size_t size, uint16_t *src)
;
int8_t vec_i8v32n_get_sum_i8(size_t size, uint8_t *src)
;
size_t vec_i8v32n_get_sum(size_t size, uint8_t *src)
;



/* assignment */

void vec_i32v8n_set_seq(size_t size, int32_t *src, int32_t start, int32_t diff)
;

