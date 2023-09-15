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


/* local */

static void dump(const char *s, __m256i current)
{
    int it;
   fputs(s, stdout);
 
    #define template_block(n) \
    { \
        it = _mm256_extract_epi32(current, n); \
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
    template_block(4)
    template_block(5)
    template_block(6)
    template_block(7)

    fputs("\n", stdout);
}
static void dump8(const char *s, __m256i current)
{

    fputs(s, stdout);

    int8_t *p = (int8_t*)&current;
    for (int i = 0; i < 16; ++i)
    {
        printf("%d,", p[i]);
    }

    fputs("\n", stdout);
}






void  vec_i16v16n_abs(size_t size, int16_t *src)
{
    size_t units = size / 16;
	
	__m256i *p = (__m256i*)src;
	__m256i umax_x_16 = _mm256_set1_epi16(UINT16_MAX);
	__m256i zero_x_16 = _mm256_setzero_si256();

	for (size_t i = 0; i < units; ++i)
	{
		__m256i it = p[i];
		
		__m256i positive_mask = _mm256_cmpgt_epi16(it, umax_x_16);
		__m256i neg_mask = _mm256_xor_si256(positive_mask, umax_x_16);
		__m256i modified = _mm256_sub_epi16(zero_x_16, it);
		it = _mm256_and_si256(it, positive_mask);
		modified = _mm256_and_si256(modified, neg_mask);
		p[i] = _mm256_or_si256(it, modified);
	}
}

void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
;



void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst)
{
	size_t units = size / 16;
	__m256i *p = (__m256i*)src1;
	__m256i *q = (__m256i*)src2;
	__m256i *r = (__m256i*)dst;

	__m256i umax_x_16 = _mm256_set1_epi16(UINT16_MAX);
	__m256i min_x_16 = _mm256_set1_epi16(INT16_MIN);
	__m256i max_x_16 = _mm256_set1_epi16(INT16_MAX);


	for (size_t i = 0; i < units; ++i)
	{
		__m256i it = _mm256_subs_epi16(q[i], p[i]);

		__m256i mask_min = _mm256_cmpeq_epi16(it, min_x_16);
		__m256i mask_positive = _mm256_cmpgt_epi16(it, umax_x_16);

		__m256i zero_x_16 = _mm256_setzero_si256();
		__m256i notmin_it = _mm256_sub_epi16(zero_x_16, it);
		notmin_it = _mm256_andnot_si256(mask_positive, notmin_it);
		it = _mm256_and_si256(it, mask_positive);
		notmin_it = _mm256_or_si256(notmin_it, it);
		notmin_it = _mm256_andnot_si256(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x_16)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm256_and_si256(mask_min, max_x_16);  // min->max
		it = _mm256_or_si256(it, notmin_it);

		r[i] = it;
	}
}


/* humming weight */

static inline size_t vec_u256n_get_humming_weight_i(size_t size, uint8_t *src)
{
#if defined(_WIN64) || defined(__X86_64__)
    size_t units = size / 8;
    uint64_t *p = (uint64_t*)src;
#else
    size_t units = size / 4;
    uint32_t *p = (uint32_t*)src;
#endif

    size_t r = 0;

    for (size_t i = 0; i < units; ++i)
    {
#if defined(_WIN64)
        r += __popcnt64(p[i]);
#elif defined(__X86_64__)
        r += __popcntq(p[i]);
#elif defined(_WIN32) && defined(_MSC_VER)
        r += __popcnt(p[i]);
#else
        r += __popcntd(p[i]);
#endif
    }

    return r;
}

size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src)
#if defined(_WIN64) || defined(__X86_64__) || defined(USE_POPCNT)
{
    return vec_u256n_get_humming_weight_i(size, src);
}
#else
{
    if (size % 64 != 0)
    {
        return vec_u256n_get_humming_weight_i(size, src);
    }

    size_t units = size / 32;

    __m256i *p = (__m256i*)src;

    __m256i rs = _mm256_setzero_si256();

    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i++];
        __m256i it2 = p[i];
        __m256i rs0 = _mm256_setzero_si256();

        __m256i one_x16 = _mm256_set1_epi8(1);

        for (int j = 0; j < 8; ++j)
        {
            __m256i tmp = _mm256_and_si256(it, one_x16);
            __m256i tmp2 = _mm256_and_si256(it2, one_x16);

            rs0 = _mm256_adds_epi8(rs0, tmp);
            rs0 = _mm256_adds_epi8(rs0, tmp2);

            it = _mm256_srli_epi32(it, 1);
            it2 = _mm256_srli_epi32(it2, 1);
        }

        __m256i mask = _mm256_set1_epi32(0x000000FF);
        __m256i mask2 = _mm256_set1_epi32(0x0000FF00);
        __m256i mask3 = _mm256_set1_epi32(0x00FF0000);
        __m256i mask4 = _mm256_set1_epi32(0xFF000000);

        __m256i masked = _mm256_and_si256(mask, rs0);
        __m256i masked2 = _mm256_and_si256(mask2, rs0);
        __m256i masked3 = _mm256_and_si256(mask3, rs0);
        __m256i masked4 = _mm256_and_si256(mask4, rs0);
        masked2 = _mm256_srli_si256(masked2, 1);
        masked3 = _mm256_srli_si256(masked3, 2);
        masked4 = _mm256_srli_si256(masked4, 3);

        masked = _mm256_add_epi32(masked, masked3);
        masked2 = _mm256_add_epi32(masked2, masked4);

        rs = _mm256_add_epi32(rs, masked);
        rs = _mm256_add_epi32(rs, masked2);
    }

    __m128i rs0 = _mm256_extracti128_si256(rs, 0);
    __m128i rs1 = _mm256_extracti128_si256(rs, 1);
    rs0 = _mm_add_epi32(rs0, rs1);

    size_t r0 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r1 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r2 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r3 = _mm_cvtsi128_si32(rs0);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}
#endif


