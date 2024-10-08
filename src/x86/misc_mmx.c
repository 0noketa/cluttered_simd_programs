#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#include "../../include/simd_tools.h"

#ifdef __3dNOW__
#define ANY_EMMS() _m_femms()
#else
#define ANY_EMMS() _m_empty()
#endif

#ifndef USE_ANDNOT
#define USE_ANDNOT
#endif

/* local */
#if 0
#include <stdio.h>
static void dump32_64(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);
 
    for (int i = 0; i < 2; ++i)
    {
        int32_t it = _mm_cvtsi64_si32(current);
        current = _mm_srli_si64(current, 32);
        ANY_EMMS();
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump16_64(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);
 
    for (int i = 0; i < 4; ++i)
    {
        int16_t it = _mm_cvtsi64_si32(current) & 0xFFFF;
        current = _mm_srli_si64(current, 16);
        ANY_EMMS();
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump8_64(const char *s, __m64 current)
{
    ANY_EMMS();
    fputs(s, stdout);

    int8_t *p = (int8_t*)&current;
    for (int i = 0; i < 8; ++i)
    {
        printf("%d,", p[i]);
    }

    fputs("\n", stdout);
}
#endif



void  vec_i16x16n_inplace_abs(size_t size, int16_t *src)
{
    size_t units = size / 4;
	__m64 *p = (__m64*)src;

	ANY_EMMS();
	__m64 max_x8 = _mm_set1_pi16(UINT16_MAX);
	__m64 minus_one_x8 = _mm_set1_pi16(-1);
	__m64 zero_x8 = _mm_setzero_si64();

	for (size_t i = 0; i < units; ++i)
	{
		__m64 it = p[i];

		__m64 mask = _mm_cmpgt_pi16(it, minus_one_x8);
		__m64 modified = _mm_sub_pi16(zero_x8, it);
		it = _mm_and_si64(it, mask);
		modified = _mm_andnot_si64(mask, modified);
		p[i] = _mm_or_si64(it, modified);
	}

	ANY_EMMS();
}




void  vec_i16x16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
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





void  vec_i16x16n_dist(size_t size, const int16_t *src1, const int16_t *src2, int16_t *dst)
{
	size_t units = size / 4;
	const __m64 *p = (const __m64*)src1;
	const __m64 *q = (const __m64*)src2;
	__m64 *r = (__m64*)dst;

	__m64 umax_x4 = _mm_set1_pi16(UINT16_MAX);
	__m64 min_x4 = _mm_set1_pi16(INT16_MIN);
	__m64 max_x4 = _mm_set1_pi16(INT16_MAX);

	ANY_EMMS();

	for (const __m64 *p_end = p + units; p < p_end;)
	{
		__m64 it = _mm_subs_pi16(*q, *p);

		__m64 mask_min = _mm_cmpeq_pi16(it, min_x4);
		__m64 mask_positive = _mm_cmpgt_pi16(it, umax_x4);

		__m64 zero_x4 = _mm_setzero_si64();
		__m64 notmin_it = _mm_sub_pi16(zero_x4, it);
		notmin_it = _mm_andnot_si64(mask_positive, notmin_it);
		it = _mm_and_si64(it, mask_positive);
		notmin_it = _mm_or_si64(notmin_it, it);
		notmin_it = _mm_andnot_si64(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x8)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm_and_si64(mask_min, max_x4);  // min->max
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
size_t vec_u256n_get_hamming_weight(size_t size, const uint8_t *src)
#ifdef USE_128BIT_UNITS
{
    size_t units = size / 8 / 2;

    const __m64 *p = (const __m64*)src;

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

    const __m64 *p = (const __m64*)src;

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

size_t vec_u256n_get_hamming_distance(size_t size, const uint8_t *src1, const uint8_t *src2)
{
#ifdef USE_128BIT_UNITS
    size_t units = size / 8 / 2;
#else
    size_t units = size / 8;
#endif

    const __m64 *p = (const __m64*)src1;
    const __m64 *q = (const __m64*)src2;

    __m64 rs = _mm_setzero_si64();

    __m64 mask = _mm_set1_pi32(0x55555555);
    __m64 mask2 = _mm_set1_pi32(0x33333333);
    __m64 mask3 = _mm_set1_pi32(0x0F0F0F0F);
    __m64 mask4 = _mm_set1_pi32(0x00FF00FF);
    __m64 mask5 = _mm_set1_pi32(0x0000FFFF);

    for (size_t i = 0; i < units; ++i)
    {
#ifdef USE_128BIT_UNITS
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];
        __m64 it_b = q[i * 2];
        __m64 it2_b = q[i * 2 + 1];

        it = _mm_xor_si64(it, it_b);
        it2 = _mm_xor_si64(it2, it2_b);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs,  mask,mask2,mask3,mask4,mask5);
#else
        __m64 it = p[i];
        __m64 it_b = q[i];

        it = _mm_xor_si64(it, it_b);

        rs = vec_u256n_get_hamming_weight_i(it, rs,  mask,mask2,mask3,mask4,mask5);
#endif
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return r;
}

size_t vec_i32x8n_get_hamming_distance(size_t size, const int32_t *src1, const int32_t *src2)
{
    size_t units = size / 2 / 2;

    const __m64 *p = (const __m64*)src1;
    const __m64 *q = (const __m64*)src2;

    // i32x2
    // counts eq (size - distance).
    __m64 rs = _mm_setzero_si64();

    __m64 one_x4 = _mm_set1_pi32(0x00000001);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];
        __m64 it_b = q[i * 2];
        __m64 it2_b = q[i * 2 + 1];

        it = _mm_cmpeq_pi32(it, it_b);
        it2 = _mm_cmpeq_pi32(it2, it2_b);
        it = _mm_and_si64(it, one_x4);
        it2 = _mm_and_si64(it2, one_x4);

        it = _mm_add_pi32(it, it2);

        rs = _mm_add_pi32(rs, it);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return size - r;
}
size_t vec_i16x16n_get_hamming_distance(size_t size, const int16_t *src1, const int16_t *src2)
{
    size_t units = size / 4 / 2;

    const __m64 *p = (const __m64*)src1;
    const __m64 *q = (const __m64*)src2;

    // i32x2
    // counts eq (size - distance).
    __m64 rs = _mm_setzero_si64();

    __m64 one_x4 = _mm_set1_pi32(0x00010001);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];
        __m64 it_b = q[i * 2];
        __m64 it2_b = q[i * 2 + 1];

        it = _mm_cmpeq_pi16(it, it_b);
        it2 = _mm_cmpeq_pi16(it2, it2_b);
        it = _mm_and_si64(it, one_x4);
        it2 = _mm_and_si64(it2, one_x4);

        it_b = _mm_slli_si64(it, 16);
        it2_b = _mm_slli_si64(it2, 16);
        it = _mm_add_pi8(it, it_b);
        it2 = _mm_add_pi8(it2, it2_b);

        it = _mm_srli_pi32(it, 16);
        it2 = _mm_srli_pi32(it2, 16);

        it = _mm_add_pi8(it, it2);

        rs = _mm_add_pi32(rs, it);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return size - r;
}
size_t vec_i8x32n_get_hamming_distance(size_t size, const int8_t *src1, const int8_t *src2)
{
    size_t units = size / 8 / 2;

    const __m64 *p = (const __m64*)src1;
    const __m64 *q = (const __m64*)src2;

    // i32x2
    // counts eq (size - distance).
    __m64 rs = _mm_setzero_si64();

    __m64 one_x8 = _mm_set1_pi32(0x01010101);

    for (size_t i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];
        __m64 it_b = q[i * 2];
        __m64 it2_b = q[i * 2 + 1];

        it = _mm_cmpeq_pi8(it, it_b);
        it2 = _mm_cmpeq_pi8(it2, it2_b);
        it = _mm_and_si64(it, one_x8);
        it2 = _mm_and_si64(it2, one_x8);

        it_b = _mm_slli_si64(it, 8);
        it2_b = _mm_slli_si64(it2, 8);
        it = _mm_add_pi8(it, it_b);
        it2 = _mm_add_pi8(it2, it2_b);

        it_b = _mm_slli_si64(it, 16);
        it2_b = _mm_slli_si64(it2, 16);
        it = _mm_add_pi8(it, it_b);
        it2 = _mm_add_pi8(it2, it2_b);

        it = _mm_srli_pi32(it, 24);
        it2 = _mm_srli_pi32(it2, 24);

        it = _mm_add_pi8(it, it2);

        rs = _mm_add_pi32(rs, it);
    }

    size_t r0 = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    size_t r1 = _mm_cvtsi64_si32(rs);

    size_t r = r0 + r1;

    ANY_EMMS();
    return size - r;
}

size_t vec_i32x8n_get_manhattan_distance(size_t size, const int32_t *src1, const int32_t *src2)
;
size_t vec_i16x16n_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2)
;
int vec_i16x16xn_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2, int32_t *dst)
;
int vec_i8x32xn_get_manhattan_distance(size_t size, const int8_t *src1, const int8_t *src2, int16_t *dst)
;




int32_t vec_i32x8n_avg(size_t size, const int32_t *src)
{
    return vec_i32x8n_sum_i32(size, src) / size;
}
double vec_i32x8n_avg_f64(size_t size, const int32_t *src)
{
    return ((double)vec_i32x8n_sum_i32(size, src) / size);
}
int16_t vec_i16x16n_avg(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum_i16(size, src) / size;
}
float vec_i16x16n_avg_f32(size_t size, const int16_t *src)
{
    return (float)((double)vec_i16x16n_sum_i16(size, src) / size);
}
int8_t vec_i8x32n_avg(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum_i32(size, src) / size;
}
float vec_i8x32n_avg_f32(size_t size, const int8_t *src)
{
    return (float)((double)vec_i8x32n_sum_i32(size, src) / size);
}


/* sum */

int32_t vec_i32x8n_sum_i32(size_t size, const int32_t *src)
{
	size_t units = size / 2 / 4;
	const __m64 *p = (const __m64*)src;

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

    int32_t r = _mm_cvtsi64_si32(rs);
    rs = _mm_srli_si64(rs, 32);
    r += _mm_cvtsi64_si32(rs);

	ANY_EMMS();
    return r;
}
uint32_t vec_u32v8n_sum_u32(size_t size, const uint32_t *src)
;

int16_t vec_i16x16n_sum_i16(size_t size, const int16_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 results2 = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];

        results = _mm_adds_pi16(results, it);
        results2 = _mm_adds_pi16(results2, it2);
    }

    results = _mm_adds_pi16(results, results2);

    results2 = _mm_srli_si64(results, 32);
    results = _mm_adds_pi16(results, results2);

    results2 = _mm_srli_si64(results, 16);
    results = _mm_adds_pi16(results, results2);

    int16_t result = _mm_cvtsi64_si32(results) & 0xFFFF;
    ANY_EMMS();
    return result;
}
int32_t vec_i16x16n_sum_i32(size_t size, const int16_t *src)
;
size_t vec_i16x16n_sum(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum_i16(size, src);
}

// returns i32x2
static __m64 vec_u16v16n_sum_i32x2(size_t size, const uint16_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;

    const __m64 mask_lower = _mm_set1_pi32(0x0000FFFF);
    __m64 results = _mm_setzero_si64();
    __m64 results2 = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];

        __m64 it_hi = _mm_slli_pi32(it, 16);
        __m64 it2_hi = _mm_slli_pi32(it2, 16);
        __m64 it_lo = _mm_and_si64(it, mask_lower);
        __m64 it2_lo = _mm_and_si64(it2, mask_lower);

        results = _mm_add_pi32(results, it_hi);
        results2 = _mm_add_pi32(results2, it2_hi);
        results = _mm_add_pi32(results, it_lo);
        results2 = _mm_add_pi32(results2, it2_lo);
    }

    return _mm_add_pi32(results, results2);
}
uint16_t vec_u16v16n_sum_u16(size_t size, const uint16_t *src)
{
    __m64 results = vec_u16v16n_sum_i32x2(size, src);
    results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
    size_t result = _mm_cvtsi64_si32(results) & 0xFFFF;

    ANY_EMMS();
    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_u16v16n_sum(size_t size, const uint16_t *src)
{
    const size_t unit_size = 0x40000000 / (UINT16_MAX + 1) * 2;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m64 results = vec_u16v16n_sum_i32x2(unit_size, src + i * unit_size);
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;
        size_t result3 = _mm_cvtsi64_si32(_mm_srli_si64(results, 32)) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
        result0 = result;
        result = result0 + result3;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m64 results = vec_u16v16n_sum_i32x2(size2, src + base);
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;
        size_t result3 = _mm_cvtsi64_si32(_mm_srli_si64(results, 32)) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
        result0 = result;
        result = result0 + result3;
        if (result < result0) result = SIZE_MAX;
    }

    ANY_EMMS();
    return result;
}

int8_t vec_i8x32n_sum_i8(size_t size, const int8_t *src)
{
    size_t units = size / 16;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 results2 = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];

        results = _mm_adds_pi8(results, it);
        results2 = _mm_adds_pi8(results2, it2);
    }

    results = _mm_adds_pi8(results, results2);

    results2 = _mm_srli_si64(results, 32);
    results = _mm_adds_pi8(results, results2);

    results2 = _mm_srli_si64(results, 16);
    results = _mm_adds_pi8(results, results2);

    results2 = _mm_srli_si64(results, 8);
    results = _mm_adds_pi8(results, results2);

    int8_t result = _mm_cvtsi64_si32(results) & 0xFF;
    ANY_EMMS();
    return result;
}

int32_t vec_i8x32n_sum_i32(size_t size, const int8_t *src)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 results2 = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 zeros = _mm_setzero_si64();
        __m64 flags = _mm_cmpgt_pi8(zeros, it);
        __m64 it2 = _mm_unpackhi_pi8(it, flags);
        it = _mm_unpacklo_pi8(it, flags);

        flags = _mm_cmpgt_pi8(zeros, it);
        __m64 flags2 = _mm_cmpgt_pi8(zeros, it2);
        __m64 it3 = _mm_unpackhi_pi16(it, flags);
        __m64 it4 = _mm_unpackhi_pi16(it2, flags2);
        it = _mm_unpacklo_pi16(it, flags);
        it2 = _mm_unpacklo_pi16(it2, flags2);

        results = _mm_add_pi32(results, it);
        results2 = _mm_add_pi32(results2, it2);
        results = _mm_add_pi32(results, it3);
        results2 = _mm_add_pi32(results2, it4);
    }

    results = _mm_add_pi32(results, results2);

    results2 = _mm_srli_si64(results, 32);
    results = _mm_add_pi32(results, results2);

    results2 = _mm_srli_si64(results, 16);
    results = _mm_add_pi32(results, results2);

    int32_t result = _mm_cvtsi64_si32(results);
    ANY_EMMS();
    return result;
}

size_t vec_i8x32n_sum(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum_i32(size, src);
}

// returns i16x4
static __m64 vec_u8v32n_sum_i16x4(size_t size, const uint8_t *src)
{
    size_t units = size / 16;
    const __m64 *p = (const void*)src;

    const __m64 mask_lower = _mm_set1_pi16(0x00FF);
    __m64 results = _mm_setzero_si64();
    __m64 results2 = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i * 2];
        __m64 it2 = p[i * 2 + 1];

        __m64 it_hi = _mm_slli_pi16(it, 8);
        __m64 it2_hi = _mm_slli_pi16(it2, 8);
        __m64 it_lo = _mm_and_si64(it, mask_lower);
        __m64 it2_lo = _mm_and_si64(it2, mask_lower);

        results = _mm_adds_pi16(results, it_hi);
        results2 = _mm_adds_pi16(results2, it2_hi);
        results = _mm_adds_pi16(results, it_lo);
        results2 = _mm_adds_pi16(results2, it2_lo);
    }

    return _mm_adds_pi16(results, results2);
}
uint8_t vec_u8v32n_sum_u8(size_t size, const uint8_t *src)
{
    __m64 results = vec_u8v32n_sum_i16x4(size, src);
    results = _mm_adds_pi16(results, _mm_srli_si64(results, 16));
    results = _mm_adds_pi16(results, _mm_srli_si64(results, 32));
    size_t result = _mm_cvtsi64_si32(results) & 0xFFFF;

    ANY_EMMS();
    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_u8v32n_sum(size_t size, const uint8_t *src)
{
    const size_t unit_size = 0x4000 / (UINT8_MAX + 1) * 4;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m64 results = vec_u8v32n_sum_i16x4(unit_size, src + i * unit_size);
        const __m64 mask_lower = _mm_set1_pi32(0x0000FFFF);
        __m64 results2 = _mm_srli_si64(results, 16);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi32(results, results2);
        results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m64 results = vec_u8v32n_sum_i16x4(size2, src + base);
        const __m64 mask_lower = _mm_set1_pi32(0x0000FFFF);
        __m64 results2 = _mm_srli_si64(results, 16);
        results = _mm_and_si64(results, mask_lower);
        results = _mm_add_pi32(results, results2);
        results = _mm_add_pi32(results, _mm_srli_si64(results, 32));
        size_t result2 = _mm_cvtsi64_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    ANY_EMMS();
    return result;
}


/* deviation sum of square */

size_t vec_i32x8n_dss_with_avg(size_t size, const int32_t *src, int32_t _avg)
;
int64_t vec_i32x8n_dss_with_avg_i64(size_t size, const int32_t *src, int32_t _avg);
size_t vec_i16x16n_dss_with_avg(size_t size, const int16_t *src, int16_t _avg)
;
int32_t vec_i16x16n_dss_with_avg_i32(size_t size, const int16_t *src, int16_t _avg);
int64_t vec_i16x16n_dss_with_avg_i64(size_t size, const int16_t *src, int16_t _avg);
uint32_t vec_u16v16n_dss_with_avg_u32(size_t size, const uint16_t *src, uint16_t _avg);
uint64_t vec_u16v16n_dss_with_avg_u64(size_t size, const uint16_t *src, uint16_t _avg);
size_t vec_i8x32n_dss_with_avg(size_t size, const int8_t *src, int8_t _avg)
{
    return vec_i8x32n_dss_with_avg_i32(size, src, _avg);
}
int16_t vec_i8x32n_dss_with_avg_i16(size_t size, const int8_t *src, int8_t _avg)
{
    return vec_i8x32n_dss_with_avg_i32(size, src, _avg);
}
int32_t vec_i8x32n_dss_with_avg_i32(size_t size, const int8_t *src, int8_t _avg)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;

    __m64 results = _mm_setzero_si64();
    __m64 avgs = _mm_set1_pi16(_avg);

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];

        __m64 zeros = _mm_setzero_si64();
        __m64 flags = _mm_cmpgt_pi8(zeros, it);
        __m64 it2 = _mm_unpackhi_pi8(it, flags);
        it = _mm_unpacklo_pi8(it, flags);

        it = _mm_subs_pi16(it, avgs);
        it2 = _mm_subs_pi16(it2, avgs);

        __m64 it_hi = _mm_mulhi_pi16(it, it);
        it = _mm_mullo_pi16(it, it);
        __m64 it2_hi = _mm_mulhi_pi16(it2, it2);
        it2 = _mm_mullo_pi16(it2, it2);
        __m64 it3 = _mm_unpackhi_pi16(it, it_hi);
        __m64 it4 = _mm_unpackhi_pi16(it2, it2_hi);
        it = _mm_unpacklo_pi16(it, it_hi);
        it2 = _mm_unpacklo_pi16(it2, it2_hi);

        it = _mm_add_pi32(it, it3);
        it2 = _mm_add_pi32(it2, it4);

        it = _mm_add_pi32(it, it2);
        results = _mm_add_pi32(results, it);
    }

    int32_t result = _mm_cvtsi64_si32(results);
    results = _mm_srli_si64(results, 32);
    int32_t result2 = _mm_cvtsi64_si32(results);
    ANY_EMMS();
    return result + result2;
}
size_t vec_u8v32n_dss_with_avg(size_t size, const uint8_t *src, uint8_t _avg)
;
uint16_t vec_u8v32n_dss_with_avg_u16(size_t size, const uint8_t *src, uint8_t _avg);
uint32_t vec_u8v32n_dss_with_avg_u32(size_t size, const uint8_t *src, uint8_t _avg);

size_t vec_i32x8n_dss(size_t size, const int32_t *src)
;
// {
//     size_t _avg = vec_i32x8n_avg(size, src);
//     return vec_i32x8n_dss_with_avg(size, src, _avg);    
// }
int64_t vec_i32x8n_dss_i64(size_t size, const int32_t *src);
size_t vec_i16x16n_dss(size_t size, const int16_t *src)
;
int32_t vec_i16x16n_dss_i32(size_t size, const int16_t *src);
int64_t vec_i16x16n_dss_i64(size_t size, const int16_t *src);
uint32_t vec_u16v16n_dss_u32(size_t size, const uint16_t *src);
uint64_t vec_u16v16n_dss_u64(size_t size, const uint16_t *src);
size_t vec_i8x32n_dss(size_t size, const int8_t *src)
{
    int8_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg_i32(size, src, _avg);
}
int16_t vec_i8x32n_dss_i16(size_t size, const int8_t *src);
int32_t vec_i8x32n_dss_i32(size_t size, const int8_t *src);
size_t vec_u8v32n_dss(size_t size, const uint8_t *src)
;
uint16_t vec_u8v32n_dss_u16(size_t size, const uint8_t *src);
uint32_t vec_u8v32n_dss_u32(size_t size, const uint8_t *src);


/* residual sum of square */

size_t vec_i32x8n_rss(size_t size, const int32_t *src, const int32_t *predicted)
;
int64_t vec_i32x8n_rss_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_rss(size_t size, const int16_t *src, const int16_t *predicted)
;
int32_t vec_i16x16n_rss_i32(size_t size, const int16_t *src, const int16_t *predicted);
int64_t vec_i16x16n_rss_i64(size_t size, const int16_t *src, const int16_t *predicted);
uint32_t vec_u16v16n_rss_u32(size_t size, const uint16_t *src, const uint16_t *predicted);
uint64_t vec_u16v16n_rss_u64(size_t size, const uint16_t *src, const uint16_t *predicted);
size_t vec_i8x32n_rss(size_t size, const uint8_t *src, const uint8_t *predicted)
{
    return vec_i8x32n_rss_i32(size, src, predicted);
}
int16_t vec_i8x32n_rss_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_rss_i32(size_t size, const int8_t *src, const int8_t *predicted)
{
    size_t units = size / 8;
    const __m64 *p = (const void*)src;
    const __m64 *q = (const void*)predicted;

    __m64 results = _mm_setzero_si64();

    for (int i = 0; i < units; ++i)
    {
        __m64 it = p[i];
        __m64 pred = q[i];

        __m64 zeros = _mm_setzero_si64();
        __m64 flags = _mm_cmpgt_pi8(zeros, it);
        __m64 flags2 = _mm_cmpgt_pi8(zeros, pred);
        __m64 it2 = _mm_unpackhi_pi8(it, flags);
        __m64 pred2 = _mm_unpackhi_pi8(pred, flags2);
        it = _mm_unpacklo_pi8(it, flags);
        pred = _mm_unpacklo_pi8(pred, flags2);

        it = _mm_subs_pi16(it, pred);
        it2 = _mm_subs_pi16(it2, pred2);

        __m64 it_hi = _mm_mulhi_pi16(it, it);
        it = _mm_mullo_pi16(it, it);
        __m64 it2_hi = _mm_mulhi_pi16(it2, it2);
        it2 = _mm_mullo_pi16(it2, it2);
        __m64 it3 = _mm_unpackhi_pi16(it, it_hi);
        __m64 it4 = _mm_unpackhi_pi16(it2, it2_hi);
        it = _mm_unpacklo_pi16(it, it_hi);
        it2 = _mm_unpacklo_pi16(it2, it2_hi);

        it = _mm_add_pi32(it, it3);
        it2 = _mm_add_pi32(it2, it4);

        it = _mm_add_pi32(it, it2);
        results = _mm_add_pi32(results, it);
    }

    int32_t result = _mm_cvtsi64_si32(results);
    results = _mm_srli_si64(results, 32);
    int32_t result2 = _mm_cvtsi64_si32(results);
    ANY_EMMS();
    return result + result2;
}
uint16_t vec_u8v32n_rss_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_rss_u32(size_t size, const uint8_t *src, const uint8_t *predicted);

/* explained sum of square */
/* sigma{ (y[i] - avg(x))^2 } */

size_t vec_i32x8n_ess(size_t size, const int32_t *src, const int32_t *predicted)
;
int64_t vec_i32x8n_ess_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_ess(size_t size, const int16_t *src, const int16_t *predicted)
;
int32_t vec_i16x16n_ess_i32(size_t size, const int16_t *src, const int16_t *predicted);
int64_t vec_i16x16n_ess_i64(size_t size, const int16_t *src, const int16_t *predicted);
uint32_t vec_u16v16n_ess_u32(size_t size, const uint16_t *src, const uint16_t *predicted);
uint64_t vec_u16v16n_ess_u64(size_t size, const uint16_t *src, const uint16_t *predicted);
size_t vec_i8x32n_ess(size_t size, const int8_t *src, const int8_t *predicted)
{
    int8_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg_i32(size, predicted, _avg);
}
int16_t vec_i8x32n_ess_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_ess_i32(size_t size, const int8_t *src, const int8_t *predicted);
size_t vec_u8v32n_ess(size_t size, const uint8_t *src, const uint8_t *predicted);
uint16_t vec_u8v32n_ess_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_ess_u32(size_t size, const uint8_t *src, const uint8_t *predicted);

