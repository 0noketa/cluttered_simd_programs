#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>
#ifdef USE_SSSE3
#include <tmmintrin.h>
#endif

#include "../../include/simd_tools.h"


/* local */
#if 0
#include <stdio.h>
static void dump32_128(const char *s, __m128i current)
{
    fputs(s, stdout);
 
    for (int i = 0; i < 4; ++i)
    {
        int32_t it = _mm_cvtsi128_si32(current);
        current = _mm_srli_si128(current, 4);
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump16_128(const char *s, __m128i current)
{
    fputs(s, stdout);
 
    for (int i = 0; i < 8; ++i)
    {
        int16_t it = _mm_cvtsi128_si32(current) & 0xFFFF;
        current = _mm_srli_si128(current, 2);
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump8_128(const char *s, __m128i current)
{

    fputs(s, stdout);

    int8_t *p = (int8_t*)&current;
    for (int i = 0; i < 16; ++i)
    {
        printf("%d,", p[i]);
    }

    fputs("\n", stdout);
}
#endif



void  vec_i16x16n_inplace_abs(size_t size, int16_t *src)
{
    size_t units = size / 8;	
	__m128i *p = (__m128i*)src;

	__m128i max_x8 = _mm_set1_epi16(UINT16_MAX);
	__m128i zero_x8 = _mm_setzero_si128();

	for (size_t i = 0; i < units; ++i)
	{
		__m128i it = p[i];
		
		__m128i neg_mask = _mm_cmplt_epi16(it, zero_x8);
		__m128i modified = _mm_sub_epi16(zero_x8, it);
		it = _mm_andnot_si128(neg_mask, it);
		modified = _mm_and_si128(modified, neg_mask);
		p[i] = _mm_or_si128(it, modified);
	}
}



void  vec_i16x16n_dist(size_t size, const int16_t *src1, const int16_t *src2, int16_t *dst)
{
	size_t units = size / 8;
	const __m128i *p = (const __m128i*)src1;
	const __m128i *q = (const __m128i*)src2;
	__m128i *r = (__m128i*)dst;

	__m128i umax_x8 = _mm_set1_epi16(UINT16_MAX);
	__m128i min_x8 = _mm_set1_epi16(INT16_MIN);
	__m128i max_x8 = _mm_set1_epi16(INT16_MAX);


	for (size_t i = 0; i < units; ++i)
	{
		__m128i it = _mm_subs_epi16(q[i], p[i]);

		__m128i mask_min = _mm_cmpeq_epi16(it, min_x8);
		__m128i mask_positive = _mm_cmpgt_epi16(it, umax_x8);

		__m128i zero_x8 = _mm_setzero_si128();
		__m128i notmin_it = _mm_sub_epi16(zero_x8, it);
		notmin_it = _mm_andnot_si128(mask_positive, notmin_it);
		it = _mm_and_si128(it, mask_positive);
		notmin_it = _mm_or_si128(notmin_it, it);
		notmin_it = _mm_andnot_si128(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x8)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm_and_si128(mask_min, max_x8);  // min->max
		it = _mm_or_si128(it, notmin_it);

		r[i] = it;
	}
}



/* hamming weight */

static inline __m128i vec_u256n_get_hamming_weight_i(__m128i it, __m128i it2, __m128i rs)
{
    __m128i mask = _mm_set1_epi32(0x55555555);
    __m128i mask2 = _mm_set1_epi32(0x33333333);
    __m128i mask3 = _mm_set1_epi32(0x0F0F0F0F);

    __m128i tmp = _mm_srli_epi32(it, 1);
    __m128i tmp2 = _mm_srli_epi32(it2, 1);
    it = _mm_and_si128(it, mask);
    it2 = _mm_and_si128(it2, mask);
    tmp = _mm_and_si128(tmp, mask);
    tmp2 = _mm_and_si128(tmp2, mask);
    it = _mm_adds_epu16(it, tmp);
    it2 = _mm_adds_epu16(it2, tmp2);

    tmp = _mm_srli_epi32(it, 2);
    tmp2 = _mm_srli_epi32(it2, 2);
    it = _mm_and_si128(it, mask2);
    it2 = _mm_and_si128(it2, mask2);
    tmp = _mm_and_si128(tmp, mask2);
    tmp2 = _mm_and_si128(tmp2, mask2);
    it = _mm_adds_epu16(it, tmp);
    it2 = _mm_adds_epu16(it2, tmp2);

    tmp = _mm_srli_epi32(it, 4);
    tmp2 = _mm_srli_epi32(it2, 4);
    it = _mm_and_si128(it, mask3);
    it2 = _mm_and_si128(it2, mask3);
    tmp = _mm_and_si128(tmp, mask3);
    tmp2 = _mm_and_si128(tmp2, mask3);
    it = _mm_add_epi32(it, tmp);
    it2 = _mm_add_epi32(it2, tmp2);

    it = _mm_add_epi32(it, it2);

    __m128i bitmask = _mm_set1_epi32(0x000000FF);
    __m128i bitmask2 = _mm_set1_epi32(0x0000FF00);
    __m128i bitmask3 = _mm_set1_epi32(0x00FF0000);
    __m128i bitmask4 = _mm_set1_epi32(0xFF000000);

    __m128i masked = _mm_and_si128(bitmask, it);
    __m128i masked2 = _mm_and_si128(bitmask2, it);
    __m128i masked3 = _mm_and_si128(bitmask3, it);
    __m128i masked4 = _mm_and_si128(bitmask4, it);
    masked2 = _mm_srli_si128(masked2, 1);
    masked3 = _mm_srli_si128(masked3, 2);
    masked4 = _mm_srli_si128(masked4, 3);

    masked = _mm_add_epi32(masked, masked3);
    masked2 = _mm_add_epi32(masked2, masked4);

    rs = _mm_add_epi32(rs, masked);
    rs = _mm_add_epi32(rs, masked2);

    return rs;
}
size_t vec_u256n_get_hamming_weight(size_t size, const uint8_t *src)
{
    size_t units = size / 16 / 2;

    const __m128i *p = (const __m128i*)src;

    __m128i rs = _mm_setzero_si128();

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs);
    }

    size_t r0 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r1 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r2 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r3 = _mm_cvtsi128_si32(rs);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}

size_t vec_u256n_get_hamming_distance(size_t size, const uint8_t *src1, const uint8_t *src2)
{
    size_t units = size / 16 / 2;

    const __m128i *p = (const __m128i*)src1;
    const __m128i *q = (const __m128i*)src2;

    __m128i rs = _mm_setzero_si128();

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];
        __m128i it_b = q[i * 2];
        __m128i it2_b = q[i * 2 + 1];

        it = _mm_xor_si128(it, it_b);
        it2 = _mm_xor_si128(it, it2_b);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs);
    }

    size_t r0 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r1 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r2 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r3 = _mm_cvtsi128_si32(rs);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}




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
    size_t units = size / 8;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    int32_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    results = _mm_srli_si128(results, 4);
    int32_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
uint32_t vec_u32v8n_sum_u32(size_t size, const uint32_t *src)
;

int16_t vec_i16x16n_sum_i16(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_adds_epi16(results, it);
        results2 = _mm_adds_epi16(results2, it2);
    }

    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi16(results, results2);

    int16_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
int32_t vec_i16x16n_sum_i32(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];
        __m128i flags = _mm_srai_epi16(it, 15);
        __m128i flags2 = _mm_srai_epi16(it2, 15);
        __m128i it3 = _mm_unpackhi_epi16(it, flags);
        __m128i it4 = _mm_unpackhi_epi16(it2, flags2);
        it = _mm_unpacklo_epi16(it, flags);
        it2 = _mm_unpacklo_epi16(it2, flags2);

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    return result;
}
size_t vec_i16x16n_sum(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum_i32(size, src);
}
uint16_t vec_u16v16n_sum_u16(size_t size, const uint16_t *src)
;
size_t vec_u16v16n_sum(size_t size, const uint16_t *src)
;

int8_t vec_i8x32n_sum_i8(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_adds_epi8(results, it);
        results2 = _mm_adds_epi8(results2, it2);
    }

    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 1);
    results = _mm_adds_epi8(results, results2);

    int8_t result = _mm_cvtsi128_si32(results) & 0xFF;
    return result;
}
int16_t vec_i8x32n_sum_i16(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];
        __m128i zeros = _mm_setzero_si128();
        __m128i flags = _mm_cmpgt_epi8(zeros, it);
        __m128i flags2 = _mm_cmpgt_epi8(zeros, it2);
        __m128i it3 = _mm_unpackhi_epi8(it, flags);
        __m128i it4 = _mm_unpackhi_epi8(it2, flags2);
        it = _mm_unpacklo_epi8(it, flags);
        it2 = _mm_unpacklo_epi8(it2, flags2);

        results = _mm_adds_epi16(results, it);
        results2 = _mm_adds_epi16(results2, it2);
    }

    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi16(results, results2);

    int16_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
int32_t vec_i8x32n_sum_i32(size_t size, const int8_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i zeros = _mm_setzero_si128();
        __m128i flags = _mm_cmpgt_epi8(zeros, it);
        __m128i it2 = _mm_unpackhi_epi8(it, flags);
        it = _mm_unpacklo_epi8(it, flags);

        flags = _mm_cmpgt_epi8(zeros, it);
        __m128i flags2 = _mm_cmpgt_epi8(zeros, it2);
        __m128i it3 = _mm_unpackhi_epi16(it, flags);
        __m128i it4 = _mm_unpackhi_epi16(it2, flags2);
        it = _mm_unpacklo_epi16(it, flags);
        it2 = _mm_unpacklo_epi16(it2, flags2);

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
        results = _mm_add_epi32(results, it3);
        results2 = _mm_add_epi32(results2, it4);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    return result;
}
size_t vec_i8x32n_sum(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum_i32(size, src);
}

// returns i16x8
static __m128i vec_u8v32n_sum_i16x8(size_t size, const uint8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    const __m128i mask_lower = _mm_set1_epi16(0x00FF);
    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        __m128i it_hi = _mm_slli_epi16(it, 8);
        __m128i it2_hi = _mm_slli_epi16(it2, 8);
        __m128i it_lo = _mm_and_si128(it, mask_lower);
        __m128i it2_lo = _mm_and_si128(it2, mask_lower);

        results = _mm_adds_epi16(results, it_hi);
        results2 = _mm_adds_epi16(results2, it2_hi);
        results = _mm_adds_epi16(results, it_lo);
        results2 = _mm_adds_epi16(results2, it2_lo);
    }

    return _mm_adds_epi16(results, results2);
}
uint8_t vec_u8v32n_sum_u8(size_t size, const uint8_t *src)
{
    __m128i results = vec_u8v32n_sum_i16x8(size, src);
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 2));
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 4));
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 8));
    size_t result = _mm_cvtsi128_si32(results) & 0xFFFF;

    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_u8v32n_sum(size_t size, const uint8_t *src)
{
    const size_t unit_size = 0x4000 / (UINT8_MAX + 1) * 8;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m128i results = vec_u8v32n_sum_i16x8(unit_size, src + i * unit_size);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_srli_si128(results, 16);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m128i results = vec_u8v32n_sum_i16x8(size2, src + base);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_srli_si128(results, 16);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

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
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i avgs = _mm_set1_epi16(_avg);

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i zeros = _mm_setzero_si128();
        __m128i flags = _mm_cmpgt_epi8(zeros, it);
        __m128i it2 = _mm_unpackhi_epi8(it, flags);
        it = _mm_unpacklo_epi8(it, flags);

        it = _mm_subs_epi16(it, avgs);
        it2 = _mm_subs_epi16(it2, avgs);

#ifdef USE_SSSE3
        it = _mm_abs_epi16(it);
        it2 = _mm_abs_epi16(it2);
#else
        flags = _mm_srai_epi16(it, 15);
        __m128i flags2 = _mm_srai_epi16(it2, 15);
        __m128i subtrahend = _mm_and_si128(it, flags);
        __m128i subtrahend2 = _mm_and_si128(it2, flags2);
        __m128i max_x16 = _mm_set1_epi16(-1);
        flags = _mm_xor_si128(flags, max_x16);
        flags2 = _mm_xor_si128(flags2, max_x16);
        it = _mm_and_si128(it, flags);
        it2 = _mm_and_si128(it2, flags2);
        it = _mm_subs_epi16(it, subtrahend);
        it2 = _mm_subs_epi16(it2, subtrahend2);
#endif
        __m128i zero_x16 = _mm_setzero_si128();
        __m128i it3 = _mm_unpackhi_epi16(it, zero_x16);
        __m128i it4 = _mm_unpackhi_epi16(it2, zero_x16);
        it = _mm_unpacklo_epi16(it, zero_x16);
        it2 = _mm_unpacklo_epi16(it2, zero_x16);

        __m128i it_b = _mm_mul_epu32(it, it);
        __m128i it2_b = _mm_mul_epu32(it2, it2);
        __m128i it3_b = _mm_mul_epu32(it3, it3);
        __m128i it4_b = _mm_mul_epu32(it4, it4);

        it = _mm_srli_si128(it, 4);
        it2 = _mm_srli_si128(it2, 4);
        it3 = _mm_srli_si128(it3, 4);
        it4 = _mm_srli_si128(it4, 4);

        it = _mm_mul_epu32(it, it);
        it2 = _mm_mul_epu32(it2, it2);
        it3 = _mm_mul_epu32(it3, it3);
        it4 = _mm_mul_epu32(it4, it4);

        it = _mm_add_epi32(it, it3);
        it2 = _mm_add_epi32(it2, it4);
        it_b = _mm_add_epi32(it_b, it3_b);
        it2_b = _mm_add_epi32(it2_b, it4_b);

        it = _mm_add_epi32(it, it2);
        it_b = _mm_add_epi32(it_b, it2_b);
        it = _mm_add_epi32(it, it_b);
        results = _mm_add_epi32(results, it);
    }

    int32_t result = _mm_cvtsi128_si32(results);
    results = _mm_srli_si128(results, 8);
    int32_t result2 = _mm_cvtsi128_si32(results);
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
    size_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg(size, src, _avg);
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
    size_t units = size / 16;
    const __m128i *p = (const void*)src;
    const __m128i *q = (const void*)predicted;

    __m128i results = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        __m128i pred = q[i];

        __m128i zeros = _mm_setzero_si128();
        __m128i flags = _mm_cmpgt_epi8(zeros, it);
        __m128i flags2 = _mm_cmpgt_epi8(zeros, pred);
        __m128i it2 = _mm_unpackhi_epi8(it, flags);
        __m128i pred2 = _mm_unpackhi_epi8(pred, flags2);
        it = _mm_unpacklo_epi8(it, flags);
        pred = _mm_unpacklo_epi8(pred, flags2);

        it = _mm_subs_epi16(it, pred);
        it2 = _mm_subs_epi16(it2, pred2);

#ifdef USE_SSSE3
        it = _mm_abs_epi16(it);
        it2 = _mm_abs_epi16(it2);
#else
        flags = _mm_srai_epi16(it, 15);
        flags2 = _mm_srai_epi16(it2, 15);
        __m128i subtrahend = _mm_and_si128(it, flags);
        __m128i subtrahend2 = _mm_and_si128(it2, flags2);
        __m128i max_x16 = _mm_set1_epi16(-1);
        flags = _mm_xor_si128(flags, max_x16);
        flags2 = _mm_xor_si128(flags2, max_x16);
        it = _mm_and_si128(it, flags);
        it2 = _mm_and_si128(it2, flags2);
        it = _mm_subs_epi16(it, subtrahend);
        it2 = _mm_subs_epi16(it2, subtrahend2);
#endif
        __m128i zero_x16 = _mm_setzero_si128();
        __m128i it3 = _mm_unpackhi_epi16(it, zero_x16);
        __m128i it4 = _mm_unpackhi_epi16(it2, zero_x16);
        it = _mm_unpacklo_epi16(it, zero_x16);
        it2 = _mm_unpacklo_epi16(it2, zero_x16);

        __m128i it_b = _mm_mul_epu32(it, it);
        __m128i it2_b = _mm_mul_epu32(it2, it2);
        __m128i it3_b = _mm_mul_epu32(it3, it3);
        __m128i it4_b = _mm_mul_epu32(it4, it4);

        it = _mm_srli_si128(it, 4);
        it2 = _mm_srli_si128(it2, 4);
        it3 = _mm_srli_si128(it3, 4);
        it4 = _mm_srli_si128(it4, 4);

        it = _mm_mul_epu32(it, it);
        it2 = _mm_mul_epu32(it2, it2);
        it3 = _mm_mul_epu32(it3, it3);
        it4 = _mm_mul_epu32(it4, it4);

        it = _mm_add_epi32(it, it3);
        it2 = _mm_add_epi32(it2, it4);
        it_b = _mm_add_epi32(it_b, it3_b);
        it2_b = _mm_add_epi32(it2_b, it4_b);

        it = _mm_add_epi32(it, it2);
        it_b = _mm_add_epi32(it_b, it2_b);
        it = _mm_add_epi32(it, it_b);
        results = _mm_add_epi32(results, it);
    }

    __m128i results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    results = _mm_srli_si128(results, 4);
    int32_t result2 = _mm_cvtsi128_si32(results);
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
    size_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg(size, predicted, _avg);
}
int16_t vec_i8x32n_ess_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_ess_i32(size_t size, const int8_t *src, const int8_t *predicted);
size_t vec_u8v32n_ess(size_t size, const uint8_t *src, const uint8_t *predicted);
uint16_t vec_u8v32n_ess_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_ess_u32(size_t size, const uint8_t *src, const uint8_t *predicted);

