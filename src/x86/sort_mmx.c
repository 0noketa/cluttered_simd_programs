#define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#include "../../include/sort.h"

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif

// #define USE_ANDNOT


/* reverse */

void vec_i32x8n_inplace_reverse(size_t size, int32_t *data)
;

void vec_i32x8n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
	vec_i32x4n_reverse(size & ~7, src, dst);
}
static void vec_i32x4n_reverse_i(size_t size, const int32_t *src, int32_t *dst)
{
	size_t units = size / 2;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	for (int i = 0; i < units / 2; ++i)
	{
		__m64 left = p[i * 2];
		__m64 right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}
}
void vec_i32x4n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
	vec_i32x4n_reverse_i(size, src, dst);
	ANY_EMMS();
}
void vec_i32x2n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
	size_t units = size / 2;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	vec_i32x4n_reverse(size, src, dst);

	if (units & 1)
	{
		__m64 it = p[units - 1];

#ifdef __3dNOW__
		it = _m_pswapd(it);
#else
		__m64 left = _mm_slli_si64(it, 32);
		__m64 right = _mm_srli_si64(it, 32);

		it = _mm_or_si64(left, right);
#endif

		q[0] = it;
	}

	ANY_EMMS();
}
// current version is slow as generic version is.
void vec_i16x16n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	vec_i16x8n_reverse(size & ~16, src, dst);
}
static void vec_i16x8n_reverse_i(size_t size, const int16_t *src, int16_t *dst)
{
	size_t units = size / 4;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	for (int i = 0; i < units / 2; ++i)
	{
		__m64 left = p[i * 2];
		__m64 right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}
}
void vec_i16x8n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	vec_i16x8n_reverse_i(size, src, dst);
	ANY_EMMS();
}
void vec_i16x4n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	size_t units = size / 4;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	vec_i16x8n_reverse(size, src, dst);

	if (units & 1)
	{
		__m64 it = p[units - 1];

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

		q[0] = it;
	}

	ANY_EMMS();
}
void vec_i8x32n_inplace_reverse(size_t size, int8_t *data)
{
	vec_i8x16n_inplace_reverse(size & ~31, data);
}
static void vec_i8x16n_inplace_reverse_i(size_t size, int8_t *data)
{
	size_t units = size / 8;
	__m64 *p = (__m64*)data;
	
	for (int i = 0; i < units / 2; ++i)
	{
		__m64 left = p[i];
		__m64 right = p[units - 1 - i];

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

		p[i] = right;
		p[units - 1 - i] = left;
	}
}
void vec_i8x16n_inplace_reverse(size_t size, int8_t *data)
{
	vec_i8x16n_inplace_reverse_i(size, data);
	ANY_EMMS();
}
void vec_i8x8n_inplace_reverse(size_t size, int8_t *data)
{
	size_t units = size / 8;
	__m64 *p = (__m64*)data;

	vec_i8x16n_inplace_reverse_i(size, data);

	if (units & 1)
	{
		__m64 it = p[units / 2];

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

		p[units / 2] = it;
	}

	ANY_EMMS();
}
void vec_i8x32n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	vec_i8x16n_reverse(size & ~31, src,dst);
}
static void vec_i8x16n_reverse_i(size_t size, const int8_t *src, int8_t *dst)
{
	size_t units = size / 8;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	for (int i = 0; i < units / 2; ++i)
	{
		__m64 left = p[i * 2];
		__m64 right = p[i * 2 + 1];

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

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}
}
void vec_i8x16n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	vec_i8x16n_reverse_i(size, src, dst);
	ANY_EMMS();
}
void vec_i8x8n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	size_t units = size / 8;
	const __m64 *p = (const __m64*)src;
	__m64 *q = (__m64*)dst;

	vec_i8x16n_reverse_i(size, src, dst);

	if (units & 1)
	{
		__m64 it = p[units - 1];

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

		q[0] = it;
	}

	ANY_EMMS();
}

void bits256n_reverse(size_t size, const uint8_t *src, uint8_t *dst)
;


/* shift */

void bits256n_shl1(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const __m64 *p = (const __m64*)src;
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
void bits256n_shl8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;
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
void bits256n_shl32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;
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

void bits256n_shr8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;
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
void bits256n_shr32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;
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

void  vec_i32x8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end);
;
void  vec_i16x16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 4;
	const __m64 *p = (const __m64*)src;

	ANY_EMMS();
	__m64 one_x4 = _mm_set1_pi16(1);
	__m64 element_x4 = _mm_set1_pi16(element);
	__m64 start0 = _mm_setzero_si64();
	__m64 end0 = _mm_setzero_si64();

	for (int i = 0; i < units; ++i)
	{
		__m64 it = p[i];
		__m64 mask_under = _mm_cmpgt_pi16(element_x4, it);
		__m64 mask_over = _mm_cmpgt_pi16(it, element_x4);
		__m64 masked_under = _mm_and_si64(mask_under, one_x4);
		__m64 masked_over = _mm_and_si64(mask_over, one_x4);
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
void  vec_i8x32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end);
;

int  vec_i32x8n_is_sorted_a(size_t size, const int32_t *src)
;
int  vec_i32x8n_is_sorted_d(size_t size, const int32_t *src)
;
int  vec_i32x8n_is_sorted(size_t size, const int32_t *src)
;

int  vec_i16x16n_is_sorted_a(size_t size, const int16_t *src)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

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
int  vec_i16x16n_is_sorted_d(size_t size, const int16_t *src)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

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
int  vec_i16x16n_is_sorted(size_t size, const int16_t *src)
{
    size_t units = size / 4;
    const __m64 *p = (const __m64*)src;

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

int  vec_i8x32n_is_sorted_a(size_t size, const int8_t *src)
;
int  vec_i8x32n_is_sorted_d(size_t size, const int8_t *src)
;
int  vec_i8x32n_is_sorted(size_t size, const int8_t *src)
;
