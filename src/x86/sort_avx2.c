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

#include "../../include/sort.h"

// #define USE_PERMUTE


/* reverse */

void vec_i32v8n_inplace_reverse(size_t size, int32_t *data)
;

void vec_i32v8n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
#ifdef USE_PERMUTE
	size_t units = size / 8;
	const __m256i *p = (const __m256i*)src;
	__m256i *q = (__m256i*)dst;

    __m256i f = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    for (int i = 0; i < units; ++i)
	{
		__m256i left = p[i];

        left = _mm256_permutevar8x32_epi32(left, f);

		q[units - 1 - i] = left;
    }
#else
	size_t units = size / 4;
	size_t units2 = units / 2;
	const __m128i *p = (const __m128i*)src;
	__m128i *q = (__m128i*)dst;

    for (int i = 0; i < units2; ++i)
	{
		__m128i left = p[i * 2];
		__m128i right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

	if (units2 & 1)
	{
		__m128i it = p[units2];

		__m128i left = _mm_slli_si128(it, 8);
		__m128i right = _mm_srli_si128(it, 8);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi64(it, 32);
		right = _mm_slli_epi64(it, 32);

		it = _mm_or_si128(left, right);

		q[units2] = it;
	}
#endif
}
// current version is slow as generic version is.
void vec_i16v16n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	size_t units = size / 8;
	size_t units2 = units / 2;
	const __m128i *p = (const __m128i*)src;
	__m128i *q = (__m128i*)dst;

    for (int i = 0; i < units2; ++i)
	{
		__m128i left = p[i * 2];
		__m128i right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

    if (units2 & 1)
	{
		__m128i it = p[units2];

		__m128i left = _mm_slli_si128(it, 8);
		__m128i right = _mm_srli_si128(it, 8);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi64(it, 32);
		right = _mm_slli_epi64(it, 32);

		it = _mm_or_si128(left, right);

		left = _mm_srli_epi32(it, 16);
		right = _mm_slli_epi32(it, 16);

		it = _mm_or_si128(left, right);

		q[units2] = it;
	}
}
void vec_i8v32n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	size_t units = size / 16;
	size_t units2 = units / 2;
	const __m128i *p = (const __m128i*)src;
	__m128i *q = (__m128i*)dst;

    for (int i = 0; i < units2; ++i)
	{
		__m128i left = p[i * 2];
		__m128i right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

    if (units2 & 1)
	{
		__m128i it = p[units2];

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

		q[units2] = it;
	}
}

void bits256n_reverse(size_t size, const uint8_t *src, uint8_t *dst)
;


/* shift */

void bits256n_shl1(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 16;
    const __m128i *p = (const __m128i*)src;
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
void bits256n_shl8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;
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
void bits256n_shl32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;
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

void bits256n_shr8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;
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
void bits256n_shr32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const __m128i *p = (const __m128i*)src;
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

void bits256n_rol1(size_t size, const uint8_t *src, uint8_t *dst)
{
    bits256n_rol8(size, src, dst);

    dst[size - 1] = src[0] >> 7;
}
void bits256n_rol8(size_t size, const uint8_t *src, uint8_t *dst)
{
    bits256n_shl8(size, src, dst);

    dst[size - 1] = src[0];
}
void bits256n_rol32(size_t size, const uint8_t *src, uint8_t *dst)
{
    bits256n_shl32(size, src, dst);

    size_t units = size / 4;
    const int32_t *p = (const int32_t*)src;
    int32_t *q = (int32_t*)dst;

    q[units - 1] = p[0];
}

void bits256n_ror8(size_t size, const uint8_t *src, uint8_t *dst)
{
    bits256n_shr8(size, src, dst);

    dst[0] = src[size - 1];
}
void bits256n_ror32(size_t size, const uint8_t *src, uint8_t *dst)
{
    bits256n_shr32(size, src, dst);

    size_t units = size / 4;
    const int32_t *p = (const int32_t*)src;
    int32_t *q = (int32_t*)dst;

    q[0] = p[units - 1];
}


/* ascendant/descendant */

void  vec_i32v8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end);
;
void  vec_i16v16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 16;
	const __m256i *p = (const __m256i*)src;

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
void  vec_i8v32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end);
;

int  vec_i32v8n_is_sorted_a(size_t size, const int32_t *src)
;
int  vec_i32v8n_is_sorted_d(size_t size, const int32_t *src)
;
int  vec_i32v8n_is_sorted(size_t size, const int32_t *src)
;

int  vec_i16v16n_is_sorted_a(size_t size, const int16_t *src)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;

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
int  vec_i16v16n_is_sorted_d(size_t size, const int16_t *src)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;

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
int  vec_i16v16n_is_sorted(size_t size, const int16_t *src)
{
    size_t units = size / 8;
    const __m128i *p = (const __m128i*)src;

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

int  vec_i8v32n_is_sorted_a(size_t size, const int8_t *src)
;
int  vec_i8v32n_is_sorted_d(size_t size, const int8_t *src)
;
int  vec_i8v32n_is_sorted(size_t size, const int8_t *src)
;


// data type: [a0, b0, ..., a1, b1, ...]
void  vec_i16v16x2n_bubblesort(size_t size, const int16_t *src, int16_t *dst)
{
	size_t units = size / 16;
	const __m256i *p = (const __m256i*)src;
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

