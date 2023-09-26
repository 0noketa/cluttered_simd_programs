#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>

#include "../../include/simd_tools.h"


/* local */
#if 0
static void dump(const char *s, __m128i current)
{
    int it;
   fputs(s, stdout);
 
    #define template_block(n) \
    { \
        __m128i shifted = _mm_srli_si128(current, n * 4); \
        it = _mm_cvtsi128_si32(shifted); \
        \
        for (int i = 0; i < 2; ++i) \
        { \
            printf("%d,", (int)(int16_t)(it & UINT16_MAX)); \
            it >>= 16; \
        } \
    }
    template_block(0)
    template_block(1)
    template_block(2)
    template_block(3)
    fputs("\n", stdout);
 }
static void dump8(const char *s, __m128i current)
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


void  vec_i16v16n_abs(size_t size, int16_t *src)
{
    size_t units = size / 8;	
	__m128i *p = (__m128i*)src;

	__m128i max_x_8 = _mm_set1_epi16(UINT16_MAX);
	__m128i zero_x_8 = _mm_setzero_si128();

	for (size_t i = 0; i < units; ++i)
	{
		__m128i it = p[i];
		
		__m128i neg_mask = _mm_cmplt_epi16(it, zero_x_8);
		__m128i modified = _mm_sub_epi16(zero_x_8, it);
		it = _mm_andnot_si128(neg_mask, it);
		modified = _mm_and_si128(modified, neg_mask);
		p[i] = _mm_or_si128(it, modified);
	}
}



void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst)
{
	size_t units = size / 8;
	__m128i *p = (__m128i*)src1;
	__m128i *q = (__m128i*)src2;
	__m128i *r = (__m128i*)dst;

	__m128i umax_x_8 = _mm_set1_epi16(UINT16_MAX);
	__m128i min_x_8 = _mm_set1_epi16(INT16_MIN);
	__m128i max_x_8 = _mm_set1_epi16(INT16_MAX);


	for (size_t i = 0; i < units; ++i)
	{
		__m128i it = _mm_subs_epi16(q[i], p[i]);

		__m128i mask_min = _mm_cmpeq_epi16(it, min_x_8);
		__m128i mask_positive = _mm_cmpgt_epi16(it, umax_x_8);

		__m128i zero_x_8 = _mm_setzero_si128();
		__m128i notmin_it = _mm_sub_epi16(zero_x_8, it);
		notmin_it = _mm_andnot_si128(mask_positive, notmin_it);
		it = _mm_and_si128(it, mask_positive);
		notmin_it = _mm_or_si128(notmin_it, it);
		notmin_it = _mm_andnot_si128(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x_8)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm_and_si128(mask_min, max_x_8);  // min->max
		it = _mm_or_si128(it, notmin_it);

		r[i] = it;
	}
}



/* hamming weight */

size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src)
{
    size_t units = size / 16 / 2;

    __m128i *p = (__m128i*)src;

    __m128i rs = _mm_setzero_si128();

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

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


size_t vec_i32v8n_get_sum(size_t size, uint32_t *src)
;
int16_t vec_i16v16n_get_sum_i16(size_t size, uint16_t *src)
;
size_t vec_i16v16n_get_sum(size_t size, uint16_t *src)
;
int8_t vec_i8v32n_get_sum_i8(size_t size, uint8_t *src)
;
size_t vec_i8v32n_get_sum(size_t size, uint8_t *src)
;
