#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

#include "../include/simd_tools.h"

#ifdef _MSC_VER
#define INIT_I16X4(n) { .n64_i16 = { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) } }
#define INIT_I8X8(n) { .n64_i8 = { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) } }
#else
#define INIT_I16X4(n) { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) }
#define INIT_I8X8(n) { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) }
#endif

/* local */

static void dump(const char *s, int16x4_t current)
{
    fputs(s, stdout);

    for (int i = 0; i < 4; ++i)
    {
        int it = (int16_t)vget_lane_s16(current, i);
        printf("%d,", it);
    }

    puts("");
}
static void dump8(const char *s, int8x8_t current)
{
    fputs(s, stdout);

    for (int i = 0; i < 8; ++i)
    {
        int it = (int16_t)vget_lane_s8(current, i);
        printf("%d,", it);
    }

    puts("");
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
    size_t units = size / 4;
    int16x4_t *p = (int16x4_t*)src;

    int16x4_t current_min = INIT_I16X4(INT16_MAX);
 
    for (size_t i = 0; i < units; ++i)
    {
        int16x4_t it = p[i];
        current_min = vmin_s16(current_min, it);
    }

    int64x1_t current_min_lo64 = vreinterpret_u64_s16(current_min);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 32);
    int16x4_t current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    current_min = vmin_s16(current_min, current_min_lo);
    current_min_lo64 = vreinterpret_u64_s16(current_min);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 16);
    current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    current_min = vmin_s16(current_min, current_min_lo);

    int16_t result = vget_lane_s16(current_min, 0);
    return result;
}

int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    size_t units = size / 4;
    int16x4_t *p = (int16x4_t*)src;

    int16x4_t current_max = INIT_I16X4(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        int16x4_t it = p[i];
        current_max = vmax_s16(current_max, it);
    }

    int64x1_t current_max_lo64 = vreinterpret_u64_s16(current_max);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 32);
    int16x4_t current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_max = vmax_s16(current_max, current_max_lo);
    current_max_lo64 = vreinterpret_u64_s16(current_max);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 16);
    current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_max = vmax_s16(current_max, current_max_lo);

    int16_t result = vget_lane_s16(current_max, 0);
    return result;
}

void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t units = size / 4;
    int16x4_t *p = (int16x4_t*)src;

    int16x4_t current_min = INIT_I16X4(INT16_MAX);
    int16x4_t current_max = INIT_I16X4(INT16_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        int16x4_t it = p[i];
        current_min = vmin_s16(current_min, it);
        current_max = vmax_s16(current_max, it);
    }

    int64x1_t current_min_lo64 = vreinterpret_u64_s16(current_min);
    int64x1_t current_max_lo64 = vreinterpret_u64_s16(current_max);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 32);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 32);
    int16x4_t current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    int16x4_t current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_min = vmin_s16(current_min, current_min_lo);
    current_max = vmax_s16(current_max, current_max_lo);
    current_min_lo64 = vreinterpret_u64_s16(current_min);
    current_max_lo64 = vreinterpret_u64_s16(current_max);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 16);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 16);
    current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_min = vmin_s16(current_min, current_min_lo);
    current_max = vmax_s16(current_max, current_max_lo);

    *out_min = vget_lane_s16(current_min, 0);
    *out_max = vget_lane_s16(current_max, 0);
}


int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    size_t units = size / 8;
    int8x8_t *p = (int8x8_t*)src;

    int8x8_t current_min = INIT_I8X8(INT8_MAX);
 
    for (size_t i = 0; i < units; ++i)
    {
        int8x8_t it = p[i];
        current_min = vmin_s8(current_min, it);
    }

    int64x1_t current_min_lo64 = vreinterpret_u64_s8(current_min);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 32);
    int8x8_t current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    current_min = vmin_s8(current_min, current_min_lo);
    current_min_lo64 = vreinterpret_u64_s8(current_min);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 16);
    current_min_lo = vreinterpret_s8_u64(current_min_lo64);
    current_min = vmin_s8(current_min, current_min_lo);
    current_min_lo64 = vreinterpret_u64_s8(current_min);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 8);
    current_min_lo = vreinterpret_s8_u64(current_min_lo64);
    current_min = vmin_s8(current_min, current_min_lo);

    int8_t result = vget_lane_s8(current_min, 0);
    return result;
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    size_t units = size / 8;
    int8x8_t *p = (int8x8_t*)src;

    int8x8_t current_max = INIT_I8X8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        int8x8_t it = p[i];
        current_max = vmax_s8(current_max, it);
    }

    int64x1_t current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 32);
    int8x8_t current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_max = vmax_s8(current_max, current_max_lo);
    current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 16);
    current_max_lo = vreinterpret_s8_u64(current_max_lo64);
    current_max = vmax_s8(current_max, current_max_lo);
    current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 8);
    current_max_lo = vreinterpret_s8_u64(current_max_lo64);
    current_max = vmax_s8(current_max, current_max_lo);

    int8_t result = vget_lane_s8(current_max, 0);
    return result;
}

void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t units = size / 8;
    int8x8_t *p = (int8x8_t*)src;

    int8x8_t current_min = INIT_I8X8(INT8_MAX);
    int8x8_t current_max = INIT_I8X8(INT8_MIN);
 
    for (size_t i = 0; i < units; ++i)
    {
        int8x8_t it = p[i];
        current_min = vmin_s8(current_min, it);
        current_max = vmax_s8(current_max, it);
    }

    int64x1_t current_min_lo64 = vreinterpret_u64_s8(current_min);
    int64x1_t current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 32);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 32);
    int8x8_t current_min_lo = vreinterpret_s16_u64(current_min_lo64);
    int8x8_t current_max_lo = vreinterpret_s16_u64(current_max_lo64);
    current_min = vmin_s8(current_min, current_min_lo);
    current_max = vmax_s8(current_max, current_max_lo);
    current_min_lo64 = vreinterpret_u64_s8(current_min);
    current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 16);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 16);
    current_min_lo = vreinterpret_s8_u64(current_min_lo64);
    current_max_lo = vreinterpret_s8_u64(current_max_lo64);
    current_min = vmin_s8(current_min, current_min_lo);
    current_max = vmax_s8(current_max, current_max_lo);
    current_min_lo64 = vreinterpret_u64_s8(current_min);
    current_max_lo64 = vreinterpret_u64_s8(current_max);
    current_min_lo64 = vshr_n_s64(current_min_lo64, 8);
    current_max_lo64 = vshr_n_s64(current_max_lo64, 8);
    current_min_lo = vreinterpret_s8_u64(current_min_lo64);
    current_max_lo = vreinterpret_s8_u64(current_max_lo64);
    current_min = vmin_s8(current_min, current_min_lo);
    current_max = vmax_s8(current_max, current_max_lo);

    *out_min = vget_lane_s8(current_min, 0);
    *out_max = vget_lane_s8(current_max, 0);
}


#if 0
void  vec_i16v16n_abs(size_t size, int16_t *src)
{
    size_t units = size / 4;
	__n64 *p = (__n64*)src;

	_m_empty();

	__n64 max_x_8 = _mm_set1_pi16(UINT16_MAX);
	__n64 minus_one_x_8 = _mm_set1_pi16(-1);
	__n64 zero_x_8;
	zero_x_8 = _mm_xor_si64(zero_x_8, zero_x_8);

	for (size_t i = 0; i < units; ++i)
	{
		__n64 it = p[i];
		__n64 mask = _mm_cmpgt_pi16(it, minus_one_x_8);
		__n64 neg_mask = _mm_xor_si64(mask, max_x_8);
		__n64 modified = _mm_sub_pi16(zero_x_8, it);
		it = _mm_and_si64(it, mask);
		modified = _mm_and_si64(modified, neg_mask);
		p[i] = _mm_or_si64(it, modified);
	}

	_m_empty();
}




void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
{
	size_t units = size / 4;
	__n64 *p = (__n64*)base;
	__n64 *q = (__n64*)target;
	__n64 *r = (__n64*)dst;

	_m_empty();

	for (size_t i = 0; i < units; ++i)
	{
		*r = _mm_subs_pi16(*q, *p);
		
		++p;
		++q;
		++r;
	}

	_m_empty();
}





void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst)
{
	size_t units = size / 4;
	__n64 *p = (__n64*)src1;
	__n64 *q = (__n64*)src2;
	__n64 *r = (__n64*)dst;

	__n64 umax_x_4 = _mm_set1_pi16(UINT16_MAX);
	__n64 min_x_4 = _mm_set1_pi16(INT16_MIN);
	__n64 max_x_4 = _mm_set1_pi16(INT16_MAX);

	_m_empty();

	for (__n64 *p_end = p + units; p < p_end;)
	{
		__n64 it = _mm_subs_pi16(*q, *p);

		__n64 mask_min = _mm_cmpeq_pi16(it, min_x_4);
		__n64 mask_positive = _mm_cmpgt_pi16(it, umax_x_4);
		__n64 mask_neg = _mm_xor_si64(mask_positive, umax_x_4);
		__n64 mask_notmin = _mm_xor_si64(mask_min, umax_x_4);

		__n64 zero_x_4;
		zero_x_4 = _mm_xor_si64(zero_x_4, zero_x_4);
		__n64 notmin_it = _mm_sub_pi16(zero_x_4, it);
		notmin_it = _mm_and_si64(notmin_it, mask_neg);
		it = _mm_and_si64(it, mask_positive);
		notmin_it = _mm_or_si64(notmin_it, it);
		notmin_it = _mm_and_si64(notmin_it, mask_notmin);

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

	_m_empty();
}


static inline void  vec_i16v16n_get_sorted_index_0(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	size_t units = size / 4;
	__n64 *p = (__n64*)src;

	__n64 one_x_4 = _mm_set1_pi16(1);
	__n64 element_x_4 = _mm_set1_pi16(element);
	__n64 start0 = _mm_setzero_si64();
	__n64 end0 = _mm_setzero_si64();

	for (int i = 0; i < units; ++i)
	{
		__n64 it = p[i];
		__n64 mask_under = _mm_cmpgt_pi16(element_x_4, it);
		__n64 mask_over = _mm_cmpgt_pi16(it, element_x_4);
		__n64 masked_under = _mm_and_si64(mask_under, one_x_4);
		__n64 masked_over = _mm_and_si64(mask_over, one_x_4);
		start0 = _mm_add_pi16(start0, masked_under);
		end0 = _mm_add_pi16(end0, masked_over);
	}
		
	__n64 start1 = _mm_srli_si64(start0, 32);
	__n64 end1 = _mm_srli_si64(end0, 32);
	start0 = _mm_add_pi16(start0, start1);
	end0 = _mm_add_pi16(end0, end1);
	start1 = _mm_srli_si64(start0, 16);
	end1 = _mm_srli_si64(end0, 16);
	start0 = _mm_add_pi16(start0, start1);
	end0 = _mm_add_pi16(end0, end1);

	*out_start = _mm_cvtsi64_si32(start0) & UINT16_MAX;
	int16_t end2 = _mm_cvtsi64_si32(end0) & UINT16_MAX;
	*out_end = size - end2;
}
#endif


void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst)
{
	size_t units = size / 2;
	int64x1_t *p = (int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	size_t i = 0;
	size_t j = units - 1;
	
	while (i < j)
	{
		uint64x1_t left = p[i];
		uint64x1_t right = p[j];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		left = vorr_u64(left_left, left_right);
		right = vorr_u64(right_left, right_right);

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[i];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		it = vorr_u32(left, right);

		p[i] = it;
	}
}
// current version is slow as generic version is.
void vec_i16v16n_get_reversed(size_t size, int16_t *src, int16_t *dst)
{
	size_t units = size / 4;
	int64x1_t *p = (int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	size_t i = 0;
	size_t j = units - 1;
	
	while (i < j)
	{
		uint64x1_t left = p[i];
		uint64x1_t right = p[j];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		uint32x2_t left32 = vreinterpret_u32_u64(vorr_u64(left_left, left_right));
		uint32x2_t right32 = vreinterpret_u32_u64(vorr_u64(right_left, right_right));

		uint32x2_t left_left32 = vshl_n_u32(left32, 16);
		uint32x2_t right_left32 = vshl_n_u32(right32, 16);
		uint32x2_t left_right32 = vshr_n_u32(left32, 16);
		uint32x2_t right_right32 = vshr_n_u32(right32, 16);

		left = vreinterpret_u64_u32(vorr_u32(left_left32, left_right32));
		right = vreinterpret_u64_u32(vorr_u32(right_left#2, right_right32));

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[i];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		uint32x2_t it32 = vreinterpret_u32_u64(vorr_u64(left, right));

		uint32x2_t left32 = vshl_n_u32(it32, 16);
		uint32x2_t right32 = vshr_n_u32(it32, 16);

		it = vreinterpret_u64_u32(vorr_u32(left32, right32));

		p[i] = it;
	}
}
void vec_i8v32n_get_reversed(size_t size, int8_t *src, int8_t *dst)
{
	size_t units = size / 8;
	int64x1_t *p = (int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	size_t i = 0;
	size_t j = units - 1;
	
	while (i < j)
	{
		uint64x1_t left = p[i];
		uint64x1_t right = p[j];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		uint32x2_t left32 = vreinterpret_u32_u64(vorr_u64(left_left, left_right));
		uint32x2_t right32 = vreinterpret_u32_u64(vorr_u64(right_left, right_right));

		uint32x2_t left_left32 = vshl_n_u32(left32, 16);
		uint32x2_t right_left32 = vshl_n_u32(right32, 16);
		uint32x2_t left_right32 = vshr_n_u32(left32, 16);
		uint32x2_t right_right32 = vshr_n_u32(right32, 16);

		uint16x4_t left16 = vreinterpret_u16_u32(vorr_u32(left_left32, left_right32));
		uint16x4_t right16 = vreinterpret_u16_u32(vorr_u32(right_left32, right_right32));

		uint16x4_t left_left16 = vshl_n_u16(left16, 8);
		uint16x4_t right_left16 = vshl_n_u16(right16, 8);
		uint16x4_t left_right16 = vshr_n_u16(left16, 8);
		uint16x4_t right_right16 = vshr_n_u16(right16, 8);

		left = vreinterpret_u64_u16(vorr_u16(left_left16, left_right16));
		right = vreinterpret_u64_u16(vorr_u16(right_left16, right_right16));

		q[i++] = right;
		q[j--] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[i];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		uint32x2_t it32 = vreinterpret_u32_u64(vorr_u64(left, right));

		uint32x2_t left32 = vshl_n_u32(it32, 16);
		uint32x2_t right32 = vshr_n_u32(it32, 16);

		uint16x4_t it16 = vreinterpret_u16_u32(vorr_u32(left32, right32));

		uint16x4_t left16 = vshl_n_u16(it16, 8);
		uint16x4_t right16 = vshr_n_u16(it16, 8);

		it = vreinterpret_u64_u16(vorr_u16(left16, right16));

		p[i] = it;
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
    uint64x1_t *p = (uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        uint64x1_t it = p[i + 1];
        uint64x1_t carried = vshl_n_u64(it, 56);
        uint64x1_t shifted = vshr_n_u64(it0, 8);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = vshr_n_u64(it0, 8);
}
void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    uint64x1_t *p = (uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        uint64x1_t it = p[i - 1];
        uint64x1_t carried = vshr_n_u64(it, 56);
        uint64x1_t shifted = vshl_n_u64(it0, 8);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = vshl_n_u64(it0, 8);
}
void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    uint64x1_t *p = (uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        uint64x1_t it = p[i + 1];
        uint64x1_t carried = vshl_n_u64(it, 32);
        uint64x1_t shifted = vshr_n_u64(it0, 32);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = vshr_n_u64(it0, 32);
}
void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    uint64x1_t *p = (uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        uint64x1_t it = p[i - 1];
        uint64x1_t carried = vshr_n_u64(it, 32);
        uint64x1_t shifted = vshl_n_u64(it0, 32);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = vshl_n_u64(it0, 32);
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

#if 0
void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	_m_empty();

	vec_i16v16n_get_sorted_index_0(size, src, element, out_start, out_end);
	
	_m_empty();
}

#endif
