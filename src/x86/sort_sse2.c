#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>

#include "../include/simd_tools.h"


/* local */

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



/* minmax */

size_t get_min_index(size_t size, int16_t *src)
;

size_t get_max_index(size_t size, int16_t *src)
;


void get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
;


int16_t vec_i16v16n_get_min(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_min = _mm_set1_epi16(INT16_MAX);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_min_epi16(current_min, it);
    }

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_min = _mm_min_epi16(current_min, current_min_lo);

    int16_t result = _mm_cvtsi128_si32(current_min) & UINT16_MAX;
    return result;
}

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_max = _mm_set1_epi16(INT16_MIN);

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_max = _mm_max_epi16(current_max, it);    }

    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_max = _mm_max_epi16(current_max, current_max_lo);

    int16_t result = _mm_cvtsi128_si32(current_max) & UINT16_MAX;
    return result;
}

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 8;
    __m128i *p = (__m128i*)src;

    __m128i current_min = _mm_set1_epi16(INT16_MAX);
    __m128i current_max = _mm_set1_epi16(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_min_epi16(current_min, it);
        current_max = _mm_max_epi16(current_max, it);
    }

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_min = _mm_min_epi16(current_min, current_min_lo);
    current_max = _mm_max_epi16(current_max, current_max_lo);

    *out_min = _mm_cvtsi128_si32(current_min) & UINT16_MAX;
    *out_max = _mm_cvtsi128_si32(current_max) & UINT16_MAX;
}


int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_min = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_add_epi8(current_min, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_min = _mm_min_epu8(current_min, it);
        current_min = _mm_sub_epi8(current_min, diff_x_16);
    }

	current_min = _mm_add_epi8(current_min, diff_x_16);

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_min_lo = _mm_srli_si128(current_min, 1);
    current_min = _mm_min_epu8(current_min, current_min_lo);

	current_min = _mm_sub_epi8(current_min, diff_x_16);

    int8_t result = _mm_cvtsi128_si32(current_min) & UINT8_MAX;
    return result;
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_max = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_max = _mm_add_epi8(current_max, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_max = _mm_max_epu8(current_max, it);
        current_max = _mm_sub_epi8(current_max, diff_x_16);
    }

	current_max = _mm_add_epi8(current_max, diff_x_16);

    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_max_lo = _mm_srli_si128(current_max, 1);
    current_max = _mm_max_epu8(current_max, current_max_lo);

	current_max = _mm_sub_epi8(current_max, diff_x_16);

    int8_t result = _mm_cvtsi128_si32(current_max) & UINT8_MAX;
    return result;
}

void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 16;
    __m128i *p = (__m128i*)src;

    __m128i diff_x_16 = _mm_set1_epi8(UINT8_MAX + INT8_MIN + 1);
    __m128i current_min = _mm_set1_epi8(INT8_MAX);
    __m128i current_max = _mm_set1_epi8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        current_min = _mm_add_epi8(current_min, diff_x_16);
        current_max = _mm_add_epi8(current_max, diff_x_16);
        it = _mm_add_epi8(it, diff_x_16);

        current_min = _mm_min_epu8(current_min, it);
        current_max = _mm_max_epu8(current_max, it);
        current_min = _mm_sub_epi8(current_min, diff_x_16);
        current_max = _mm_sub_epi8(current_max, diff_x_16);
    }

	current_min = _mm_add_epi8(current_min, diff_x_16);
	current_max = _mm_add_epi8(current_max, diff_x_16);

    __m128i current_min_lo = _mm_srli_si128(current_min, 8);
    __m128i current_max_lo = _mm_srli_si128(current_max, 8);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 4);
    current_max_lo = _mm_srli_si128(current_max, 4);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 2);
    current_max_lo = _mm_srli_si128(current_max, 2);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);
    current_min_lo = _mm_srli_si128(current_min, 1);
    current_max_lo = _mm_srli_si128(current_max, 1);
    current_min = _mm_min_epu8(current_min, current_min_lo);
    current_max = _mm_max_epu8(current_max, current_max_lo);

	current_min = _mm_sub_epi8(current_min, diff_x_16);
	current_max = _mm_sub_epi8(current_max, diff_x_16);

    *out_min = _mm_cvtsi128_si32(current_min) & UINT8_MAX;
    *out_max  = _mm_cvtsi128_si32(current_max) & UINT8_MAX;
}


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


void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst)
{
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
    size_t units = size / 16;
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
    size_t units = size / 16;
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
    size_t units = size / 16;
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
    size_t units = size / 16;
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

size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src)
{
    size_t units = size / 16;

    __m128i *p = (__m128i*)src;

    __m128i rs = _mm_setzero_si128();

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i++];
        __m128i it2 = p[i];
        __m128i rs0 = _mm_setzero_si128();

        __m128i one_x16 = _mm_set1_epi8(1);

        for (int j = 0; j < 8; ++j)
        {
            __m128i tmp = _mm_and_si128(it, one_x16);
            __m128i tmp2 = _mm_and_si128(it2, one_x16);

            rs0 = _mm_adds_epi8(rs0, tmp);
            rs0 = _mm_adds_epi8(rs0, tmp2);

            it = _mm_srli_epi32(it, 1);
            it2 = _mm_srli_epi32(it2, 1);
         }

        __m128i mask = _mm_set1_epi32(0x000000FF);
        __m128i mask2 = _mm_set1_epi32(0x0000FF00);
        __m128i mask3 = _mm_set1_epi32(0x00FF0000);
        __m128i mask4 = _mm_set1_epi32(0xFF000000);

        __m128i masked = _mm_and_si128(mask, rs0);
        __m128i masked2 = _mm_and_si128(mask2, rs0);
        __m128i masked3 = _mm_and_si128(mask3, rs0);
        __m128i masked4 = _mm_and_si128(mask4, rs0);
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


/* sorted arrays */

void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 8;
	__m128i *p = (__m128i*)src;

	__m128i one_x_8 = _mm_set1_epi16(1);
	__m128i element_x_8 = _mm_set1_epi16(element);
	__m128i start0 = _mm_setzero_si128();
	__m128i end0 = _mm_setzero_si128();

	for (int i = 0; i < units; ++i)
	{
		__m128i it = p[i];
		__m128i mask_under = _mm_cmpgt_epi16(element_x_8, it);
		__m128i mask_over = _mm_cmpgt_epi16(it, element_x_8);
		__m128i masked_under = _mm_and_si128(mask_under, one_x_8);
		__m128i masked_over = _mm_and_si128(mask_over, one_x_8);
		start0 = _mm_add_epi16(start0, masked_under);
		end0 = _mm_add_epi16(end0, masked_over);
	}
		
	__m128i start1 = _mm_srli_si128(start0, 8);
	__m128i end1 = _mm_srli_si128(end0, 8);
	start0 = _mm_add_epi16(start0, start1);
	end0 = _mm_add_epi16(end0, end1);
	start1 = _mm_srli_si128(start0, 4);
	end1 = _mm_srli_si128(end0, 4);
	start0 = _mm_add_epi16(start0, start1);
	end0 = _mm_add_epi16(end0, end1);
	start1 = _mm_srli_si128(start0, 2);
	end1 = _mm_srli_si128(end0, 2);
	start0 = _mm_add_epi16(start0, start1);
	end0 = _mm_add_epi16(end0, end1);
	
	*out_start = _mm_cvtsi128_si32(start0) & UINT16_MAX;
	int16_t end2 = _mm_cvtsi128_si32(end0) & UINT16_MAX;
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

/*
int  vec_i16v16n_is_sortable(size_t size, int16_t *src)
{
	return size <= (1 << 15) - 16 && ((uintptr_t)size % 8) == 0 && ((uintptr_t)src % 16) == 0;
}


// takes x10 more than qsort
// it means unique list can not help
void  vec_i16v16n_sort(size_t size, int16_t *src, int16_t *dst)
{
	for (size_t i = 0; i < size; ++i)
	{
		int_fast16_t it = src[i];
		int16_t start, end;
		
		vec_i16v16n_get_sorted_index(size, src, it, &start, &end);

		for (size_t j = start; j < end; ++j)
		{
			dst[j] = it;
		}
	}
}
*/

// data type: [a0, b0, ..., a1, b1, ...]
void  vec_i16v8x2n_bubblesort(size_t size, int16_t *src, int16_t *dst)
{
	size_t units = size / 8;
	__m128i *p = (__m128i*)src;
	__m128i *q = (__m128i*)dst;

	int modified = 0;
	do
	{
		__m128i modified0 = _mm_setzero_si128();

		__m128i it = p[0];
		for (size_t i = 0; i < units - 1; ++i)
		{
			__m128i nxt = p[i + 1];
			
			__m128i it2 = _mm_min_epi16(it, nxt);
			__m128i nxt2 = _mm_max_epi16(it, nxt);
			
			__m128i modified1 = _mm_xor_si128(it, it2);
			__m128i modified2 = _mm_xor_si128(nxt, nxt2);
			modified0 = _mm_or_si128(modified0, modified1);
			modified0 = _mm_or_si128(modified0, modified2);

			q[i] = it2;
			it = nxt2;
		}
		q[units - 1] = it;
		p = q;

		__m128i modified1 = _mm_srli_si128(modified0, 8);
		modified0 = _mm_or_si128(modified0, modified1);
		modified1 = _mm_srli_si128(modified0, 4);
		modified0 = _mm_or_si128(modified0, modified1);
		
		modified = _mm_cvtsi128_si32(modified0);
	}
	while (modified);
}
