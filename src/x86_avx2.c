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



/* minmax */

size_t get_min_index(size_t size, int16_t *src)
;

size_t get_max_index(size_t size, int16_t *src)
;


void get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
;


int16_t vec_i16v16n_get_min(size_t size, int16_t *src)
{
    size_t units = size / 16;
    __m256i current_min;
    __m256i *p = (__m256i*)src;

    current_min = _mm256_set1_epi16(INT16_MAX);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_min = _mm256_min_epi16(current_min, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_min = _mm256_min_epi16(current_min, current_min_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    result_min0 = _mm_min_epi16(result_min0, result_min1);

    int16_t result = _mm_cvtsi128_si32(result_min0) & UINT16_MAX;
    return result;
}

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 16;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

    current_max = _mm256_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_max = _mm256_max_epi16(current_max, it);
    }

    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_max = _mm256_max_epi16(current_max, current_max_lo);

    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_max0 = _mm_max_epi16(result_max0, result_max1);

    int16_t result = _mm_cvtsi128_si32(result_max0) & UINT16_MAX;
    return result;
}

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 16;
    __m256i current_min;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

    current_min = _mm256_set1_epi16(INT16_MAX);
    current_max = _mm256_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];
        current_min = _mm256_min_epi16(current_min, it);
        current_max = _mm256_max_epi16(current_max, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_min = _mm256_min_epi16(current_min, current_min_lo);
    current_max = _mm256_max_epi16(current_max, current_max_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_min0 = _mm_min_epi16(result_min0, result_min1);
    result_max0 = _mm_max_epi16(result_max0, result_max1);

    *out_min = _mm_cvtsi128_si32(result_min0) & UINT16_MAX;
    *out_max = _mm_cvtsi128_si32(result_max0) & UINT16_MAX;
 }


int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 32;
    __m256i current_min;
    __m256i *p = (__m256i*)src;

    current_min = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_min = _mm256_min_epi8(current_min, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_min_lo = _mm256_srli_si256(current_min, 1);
    current_min = _mm256_min_epi8(current_min, current_min_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    result_min0 = _mm_min_epi8(result_min0, result_min1);

    int8_t result = _mm_cvtsi128_si32(result_min0) & UINT8_MAX;
    return result;
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 32;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

    current_max = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_max = _mm256_max_epi8(current_max, it);
    }

    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_max_lo = _mm256_srli_si256(current_max, 1);
    current_max = _mm256_max_epi8(current_max, current_max_lo);

    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_max0 = _mm_max_epi8(result_max0, result_max1);

    int8_t result = _mm_cvtsi128_si32(result_max0) & UINT8_MAX;
    return result;
}
/*stub*/
void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 32;
    __m256i current_min;
    __m256i current_max;
    __m256i *p = (__m256i*)src;

    current_min = _mm256_set1_epi8(INT8_MAX);
    current_max = _mm256_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i];

        current_min = _mm256_min_epi8(current_min, it);
        current_max = _mm256_max_epi8(current_max, it);
    }

    __m256i current_min_lo = _mm256_srli_si256(current_min, 8);
    __m256i current_max_lo = _mm256_srli_si256(current_max, 8);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 4);
    current_max_lo = _mm256_srli_si256(current_max, 4);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 2);
    current_max_lo = _mm256_srli_si256(current_max, 2);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);
    current_min_lo = _mm256_srli_si256(current_min, 1);
    current_max_lo = _mm256_srli_si256(current_max, 1);
    current_min = _mm256_min_epi8(current_min, current_min_lo);
    current_max = _mm256_max_epi8(current_max, current_max_lo);

    __m128i result_min0 = _mm256_extracti128_si256(current_min, 0);
    __m128i result_min1 = _mm256_extracti128_si256(current_min, 1);
    __m128i result_max0 = _mm256_extracti128_si256(current_max, 0);
    __m128i result_max1 = _mm256_extracti128_si256(current_max, 1);
    result_min0 = _mm_min_epi8(result_min0, result_min1);
    result_max0 = _mm_max_epi8(result_max0, result_max1);

    *out_min = _mm_cvtsi128_si32(result_min0) & UINT8_MAX;
    *out_max = _mm_cvtsi128_si32(result_max0) & UINT8_MAX;
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


void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst)
{
#ifdef USE_PERMUTE
	size_t units = size / 8;
	__m256i *p = (__m256i*)src;
	__m256i *q = (__m256i*)dst;

	size_t i = 0;
	size_t j = units - 1;
	
    while (i < units)
    {
        __m256i left = p[i++];

        left = _mm256_permutevar8x32_epi32(left, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7));

        _mm256_store_si256(q + j--, left);
    }
#else
	size_t units = size / 4;
	__m128i *p = (__m128i*)src;
	__m128i *q = (__m128i*)dst;

	size_t i = 0;
	size_t j = units - 1;
	
	while (i < units)
    {
		__m128i left = p[i++];
		__m128i right = p[i++];

		__m128i left_left = _mm_slli_si128(left, 8);
		__m128i right_left = _mm_slli_si128(right, 8);
		__m128i left_right = _mm_srli_si128(left, 8);
		__m128i right_right = _mm_srli_si128(right, 8);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi64(left, 32);
		right_left = _mm_srli_epi64(right, 32);
		left_right = _mm_slli_epi64(left, 32);
		right_right = _mm_slli_epi64(right, 32);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		// q[j--] = left;
		// q[j--] = right;
        _mm_store_si128(q + j--, left);
        _mm_store_si128(q + j--, right);
	}
#endif
}
// current version is slow as generic version is.
void vec_i16v16n_get_reversed(size_t size, int16_t *src, int16_t *dst)
{
	size_t units = size / 8;
	__m128i *p = (__m128i*)src;
	__m128i *q = (__m128i*)dst;

	size_t i = 0;
	size_t j = units - 1;

	while (i < j)
	{
		__m128i left = p[i];
		__m128i right = p[j];

		// every var name means bytes' order
		// mmx has not int64

		__m128i left_left = _mm_slli_si128(left, 8);
		__m128i right_left = _mm_slli_si128(right, 8);
		__m128i left_right = _mm_srli_si128(left, 8);
		__m128i right_right = _mm_srli_si128(right, 8);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi64(left, 32);
		right_left = _mm_srli_epi64(right, 32);
		left_right = _mm_slli_epi64(left, 32);
		right_right = _mm_slli_epi64(right, 32);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi32(left, 16);
		right_left = _mm_srli_epi32(right, 16);
		left_right = _mm_slli_epi32(left, 16);
		right_right = _mm_slli_epi32(right, 16);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		__m128i it = p[i];

		__m128i left = _mm_slli_si128(it, 8);
		__m128i right = _mm_srli_si128(it, 8);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi32(it, 16);
		right = _mm_slli_epi32(it, 16);

		it = _mm_or_si128(left, right);

		q[i] = it;
	}
}
void vec_i8v32n_get_reversed(size_t size, int8_t *src, int8_t *dst)
{
	size_t units = size / 16;
	__m128i *p = (__m128i*)src;
	__m128i *q = (__m128i*)dst;

	size_t i = 0;
	size_t j = units - 1;

	while (i < j)
	{
		__m128i left = p[i];
		__m128i right = p[j];

		// every var name means bytes' order
		// mmx has not int64

		__m128i left_left = _mm_slli_si128(left, 8);
		__m128i right_left = _mm_slli_si128(right, 8);
		__m128i left_right = _mm_srli_si128(left, 8);
		__m128i right_right = _mm_srli_si128(right, 8);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi64(left, 32);
		right_left = _mm_srli_epi64(right, 32);
		left_right = _mm_slli_epi64(left, 32);
		right_right = _mm_slli_epi64(right, 32);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi32(left, 16);
		right_left = _mm_srli_epi32(right, 16);
		left_right = _mm_slli_epi32(left, 16);
		right_right = _mm_slli_epi32(right, 16);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		left_left = _mm_srli_epi16(left, 8);
		right_left = _mm_srli_epi16(right, 8);
		left_right = _mm_slli_epi16(left, 8);
		right_right = _mm_slli_epi16(right, 8);

		left = _mm_or_si128(left_left, left_right);
		right = _mm_or_si128(right_left, right_right);

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		__m128i it = p[i];

		__m128i left = _mm_slli_si128(it, 8);
		__m128i right = _mm_srli_si128(it, 8);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi64(it, 32);
		right = _mm_slli_epi64(it, 32);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi32(it, 16);
		right = _mm_slli_epi32(it, 16);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi16(it, 8);
		right = _mm_slli_epi16(it, 8);

		it = _mm_or_si128(left, right);

		q[i] = it;
	}
}




/* shift */

void vec_u256n_shl1(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i mask_high = _mm_set1_epi8(UINT8_MAX - 1);
    __m128i mask_low = _mm_set1_epi8(1);
    __m128i mask_carry = _mm_set_epi32(0x01000000, 0, 0, 0);
    __m128i it = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m128i next = p[i + 1];
        __m128i shifted = _mm_slli_epi32(it, 1);
        __m128i shifted2 = _mm_srli_epi32(it, 7);
        shifted = _mm_and_si128(shifted, mask_high);
        shifted2 = _mm_and_si128(shifted2, mask_low);
        __m128i shifted3 = _mm_slli_si128(next, 15);
        shifted2 = _mm_srli_si128(shifted2, 1);
        shifted3 = _mm_srli_epi32(shifted3, 7);

        __m128i it2 = _mm_or_si128(shifted, shifted2);
        shifted3 = _mm_and_si128(shifted3, mask_carry);
        it2 = _mm_or_si128(it2, shifted3);

        q[i] = it2;

        it = next;
    }

    {
        __m128i shifted = _mm_slli_epi32(it, 1);
        __m128i shifted2 = _mm_srli_epi32(it, 7);
        shifted = _mm_and_si128(shifted, mask_high);
        shifted2 = _mm_and_si128(shifted2, mask_low);
        shifted2 = _mm_srli_si128(shifted2, 1);

        q[units - 1] = _mm_or_si128(shifted, shifted2);
    }
}
void vec_u256n_shl8(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m128i it = p[i + 1];
        __m128i carried = _mm_slli_si128(it, 15);
        __m128i shifted = _mm_srli_si128(it0, 1);
        shifted = _mm_or_si128(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = _mm_srli_si128(it0, 1);
}
void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        __m128i it = p[i - 1];
        __m128i carried = _mm_srli_si128(it, 15);
        __m128i shifted = _mm_slli_si128(it0, 1);
        shifted = _mm_or_si128(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = _mm_slli_si128(it0, 1);
}
void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m128i it = p[i + 1];
        __m128i carried = _mm_slli_si128(it, 12);
        __m128i shifted = _mm_srli_si128(it0, 4);
        shifted = _mm_or_si128(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = _mm_srli_si128(it0, 4);
}
void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    __m128i *p = (__m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        __m128i it = p[i - 1];
        __m128i carried = _mm_srli_si128(it, 12);
        __m128i shifted = _mm_slli_si128(it0, 4);
        shifted = _mm_or_si128(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = _mm_slli_si128(it0, 1);
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


/* sorted arrays */

void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 16;
	__m256i *p = (__m256i*)src;

	__m256i one_x_16 = _mm256_set1_epi16(1);
	__m256i element_x_16 = _mm256_set1_epi16(element);
	__m256i start0 = _mm256_setzero_si256();
	__m256i end0 = _mm256_setzero_si256();

	for (int i = 0; i < units; ++i)
	{
		__m256i it = p[i];
		__m256i mask_under = _mm256_cmpgt_epi16(element_x_16, it);
		__m256i mask_over = _mm256_cmpgt_epi16(it, element_x_16);
		__m256i masked_under = _mm256_and_si256(mask_under, one_x_16);
		__m256i masked_over = _mm256_and_si256(mask_over, one_x_16);
		start0 = _mm256_adds_epi16(start0, masked_under);
		end0 = _mm256_adds_epi16(end0, masked_over);
	}

	__m256i start1 = _mm256_srli_si256(start0, 8);
	__m256i end1 = _mm256_srli_si256(end0, 8);
	start0 = _mm256_adds_epi16(start0, start1);
	end0 = _mm256_adds_epi16(end0, end1);
	start1 = _mm256_srli_si256(start0, 4);
	end1 = _mm256_srli_si256(end0, 4);
	start0 = _mm256_adds_epi16(start0, start1);
	end0 = _mm256_adds_epi16(end0, end1);
	start1 = _mm256_srli_si256(start0, 2);
	end1 = _mm256_srli_si256(end0, 2);
	start0 = _mm256_adds_epi16(start0, start1);
	end0 = _mm256_adds_epi16(end0, end1);
	
    __m128i result_start0 = _mm256_extracti128_si256(start0, 0);
    __m128i result_start1 = _mm256_extracti128_si256(start0, 1);
    __m128i result_end0 = _mm256_extracti128_si256(end0, 0);
    __m128i result_end1 = _mm256_extracti128_si256(end0, 1);
    result_start0 = _mm_adds_epi16(result_start0, result_start1);
    result_end0 = _mm_adds_epi16(result_end0, result_end1);
	
	*out_start = _mm_cvtsi128_si32(result_start0) & UINT16_MAX;
	int16_t end2 = _mm_cvtsi128_si32(result_end0) & UINT16_MAX;
	*out_end = size - end2;
}


int  vec_i16v16n_is_sorted_a(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i it0 = _mm_set1_epi16(INT16_MIN);
    __m128i cond0 = _mm_set1_epi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i it_left = _mm_slli_si128(it, 2);
        it0 = _mm_srli_si128(it0, 14);
        it_left = _mm_or_si128(it0, it_left);

        __m128i cond = _mm_cmpgt_epi16(it_left, it);
        
        cond0 = _mm_andnot_si128(cond, cond0);
        it0 = it;
    }

    __m128i cond_right = _mm_srli_si128(cond0, 8);
    cond0 = _mm_and_si128(cond0, cond_right);
    cond_right = _mm_srli_si128(cond0, 4);
    cond0 = _mm_and_si128(cond0, cond_right);
    cond_right = _mm_srli_si128(cond0, 2);
    cond0 = _mm_and_si128(cond0, cond_right);

    int result = _mm_cvtsi128_si32(cond0) & UINT16_MAX;
    return !!result;
}
int  vec_i16v16n_is_sorted_d(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i it0 = _mm_set1_epi16(INT16_MAX);
    __m128i cond0 = _mm_set1_epi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i it_left = _mm_slli_si128(it, 2);
        it0 = _mm_srli_si128(it0, 14);
        it_left = _mm_or_si128(it0, it_left);

        __m128i cond = _mm_cmpgt_epi16(it, it_left);

        cond0 = _mm_andnot_si128(cond, cond0);
        it0 = it;
    }

    __m128i cond_right = _mm_srli_si128(cond0, 8);
    cond0 = _mm_and_si128(cond0, cond_right);
    cond_right = _mm_srli_si128(cond0, 4);
    cond0 = _mm_and_si128(cond0, cond_right);
    cond_right = _mm_srli_si128(cond0, 2);
    cond0 = _mm_and_si128(cond0, cond_right);

    int result = _mm_cvtsi128_si32(cond0) & UINT16_MAX;
    return !!result;
}
int  vec_i16v16n_is_sorted(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i it_a0 = _mm_set1_epi16(INT16_MIN);
    __m128i it_d0 = _mm_set1_epi16(INT16_MAX);
    __m128i cond_a0 = _mm_set1_epi16(-1);
    __m128i cond_d0 = _mm_set1_epi16(-1);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i it_left = _mm_slli_si128(it, 2);
        it_a0 = _mm_srli_si128(it_a0, 14);
        it_d0 = _mm_srli_si128(it_d0, 14);
        __m128i it_left_a = _mm_or_si128(it_a0, it_left);
        __m128i it_left_d = _mm_or_si128(it_d0, it_left);

        __m128i cond_a = _mm_cmpgt_epi16(it_left_a, it);
        __m128i cond_d = _mm_cmpgt_epi16(it, it_left_d);

        cond_a0 = _mm_andnot_si128(cond_a, cond_a0);
        cond_d0 = _mm_andnot_si128(cond_d, cond_d0);
        it_a0 = it;
        it_d0 = it;
    }

    __m128i cond_a_right = _mm_srli_si128(cond_a0, 8);
    __m128i cond_d_right = _mm_srli_si128(cond_d0, 8);
    cond_a0 = _mm_and_si128(cond_a0, cond_a_right);
    cond_d0 = _mm_and_si128(cond_d0, cond_d_right);
    cond_a_right = _mm_srli_si128(cond_a0, 4);
    cond_d_right = _mm_srli_si128(cond_d0, 4);
    cond_a0 = _mm_and_si128(cond_a0, cond_a_right);
    cond_d0 = _mm_and_si128(cond_d0, cond_d_right);
    cond_a_right = _mm_srli_si128(cond_a0, 2);
    cond_d_right = _mm_srli_si128(cond_d0, 2);
    cond_a0 = _mm_and_si128(cond_a0, cond_a_right);
    cond_d0 = _mm_and_si128(cond_d0, cond_d_right);

    int result_a = _mm_cvtsi128_si32(cond_a0) & UINT16_MAX;
    int result_d = _mm_cvtsi128_si32(cond_d0) & UINT16_MAX;
    return result_a || result_d;
}


// data type: [a0, b0, ..., a1, b1, ...]
void  vec_i16v16x2n_bubblesort(size_t size, int16_t *src, int16_t *dst)
{
	size_t units = size / 16;
	__m256i *p = (__m256i*)src;
	__m256i *q = (__m256i*)dst;

	int modified = 0;
	do
	{
		__m256i modified0 = _mm256_setzero_si256();

		__m256i it = p[0];
		for (size_t i = 0; i < units - 1; ++i)
		{
			__m256i nxt = p[i + 1];
			
			__m256i it2 = _mm256_min_epi16(it, nxt);
			__m256i nxt2 = _mm256_max_epi16(it, nxt);
			
			__m256i modified1 = _mm256_xor_si256(it, it2);
			__m256i modified2 = _mm256_xor_si256(nxt, nxt2);
			modified0 = _mm256_or_si256(modified0, modified1);
			modified0 = _mm256_or_si256(modified0, modified2);

			q[i] = it2;
			it = nxt2;
		}
		q[units - 1] = it;
		p = q;

		__m128i modified0_0 = _mm256_extracti128_si256(modified0, 0);
		__m128i modified0_1 = _mm256_extracti128_si256(modified0, 1);

		__m128i modified1_0 = _mm_srli_si128(modified0_0, 8);
		__m128i modified1_1 = _mm_srli_si128(modified0_1, 8);
		modified0_0 = _mm_or_si128(modified0_0, modified1_0);
		modified0_1 = _mm_or_si128(modified0_1, modified1_1);
		modified1_0 = _mm_srli_si128(modified0_0, 4);
		modified1_1 = _mm_srli_si128(modified0_1, 4);
		modified0_0 = _mm_or_si128(modified0_0, modified1_0);
		modified0_1 = _mm_or_si128(modified0_1, modified1_1);
		
		modified = _mm_cvtsi128_si32(modified0_0);
		modified |= _mm_cvtsi128_si32(modified0_1);
	}
	while (modified);
}
