#include <stddef.h>
#include <stdint.h>

#include "../../include/simd_tools.h"

#ifdef USE_LUT
#include "./hw_16bit_lut.c"
#endif

/* local */


// -O2 is faster as mmx/sse2 versions on Atom N2700
void  vec_i16x16n_inplace_abs(size_t size, int16_t *src)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int16_t it = src[i];

		if (it < 0) src[i] = -it;
	}
}

void  vec_i16x16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int32_t it = target[i] - base[i];

		if (it < INT16_MIN) it = INT16_MIN;
		if (it > INT16_MAX) it = INT16_MAX;

		dst[i] = it;
	}
}

void  vec_i16x16n_parallel_dist(size_t size, const int16_t *src1, const int16_t *src2, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int32_t it = src2[i] - src1[i];

		if (it <= INT16_MIN)
		{
			it = INT16_MIN;
		}
		else  if (it > INT16_MAX)
		{
			it = INT16_MAX;
		}
		else if (it < 0)
		{
			it = -it;
		}

		dst[i] = it;
	}
}


/* hamming weight */

size_t vec_u256n_get_hamming_weight(size_t size, const uint8_t *src)
{
    size_t units = size / 4;
    const uint32_t *p = (const void*)src;
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < units; ++i)
    {
        uint32_t it = p[i];
#ifdef USE_LUT
        r += hw_16bit_lut_[it & 0xFFFF] + hw_16bit_lut_[it >> 16];
#else
        it = (it & 0x55555555) + ((it >> 1) & 0x55555555);
        it = (it & 0x33333333) + ((it >> 2) & 0x33333333);
        it = (it & 0x0F0F0F0F) + ((it >> 4) & 0x0F0F0F0F);
        it = (it & 0x00FF00FF) + ((it >> 8) & 0x00FF00FF);

        r += it & 0xFF;
        r += (it >> 16) & 0xFF;
#endif
    }

    return r;
}
size_t vec_u256n_get_hamming_distance(size_t size, const uint8_t *src1, const uint8_t *src2)
{
    size_t units = size / 4;
    const uint32_t *p = (const void*)src1;
    uint32_t *q = (void*)src2;
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < units; ++i)
    {
        uint32_t it = p[i] ^ q[i];

#ifdef USE_LUT
        r += hw_16bit_lut_[it & 0xFFFF] + hw_16bit_lut_[it >> 16];
#else
        it = (it & 0x55555555) + ((it >> 1) & 0x55555555);
        it = (it & 0x33333333) + ((it >> 2) & 0x33333333);
        it = (it & 0x0F0F0F0F) + ((it >> 4) & 0x0F0F0F0F);
        it = (it & 0x00FF00FF) + ((it >> 8) & 0x00FF00FF);

        r += it & 0xFF;
        r += (it >> 16) & 0xFF;
#endif
    }

    return r;
}
size_t vec_i32x8n_get_hamming_distance(size_t size, const int32_t *src1, const int32_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        r += src1[i] != src2[i];
    }

    return r;
}
size_t vec_i16x16n_get_hamming_distance(size_t size, const int16_t *src1, const int16_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        r += src1[i] != src2[i];
    }

    return r;
}
size_t vec_i8x32n_get_hamming_distance(size_t size, const int8_t *src1, const int8_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        r += src1[i] != src2[i];
    }

    return r;
}

size_t vec_i32x8n_get_manhattan_distance(size_t size, const int32_t *src1, const int32_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        int32_t it = src1[i];
        int32_t it2 = src2[i];

        r += it > it2 ? it - it2 : it2 - it;
    }

    return r;
}
size_t vec_i16x16n_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src1[i];
        int16_t it2 = src2[i];

        r += it > it2 ? it - it2 : it2 - it;
    }

    return r;
}
size_t vec_i8x32n_get_manhattan_distance(size_t size, const int8_t *src1, const int8_t *src2)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src1[i];
        int8_t it2 = src2[i];

        r += it > it2 ? it - it2 : it2 - it;
    }

    return r;
}

int vec_i16x16xn_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2, int32_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int32_t r = 0;
        for (int j = 0; j < 16; ++j)
        {
            int16_t it = src1[i * 16 + j];
            int16_t it2 = src2[i * 16 + j];

            r += it > it2 ? it - it2 : it2 - it;
        }

        dst[i] = r;
    }

    return 1;
}
int vec_i8x32xn_get_manhattan_distance(size_t size, const int8_t *src1, const int8_t *src2, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t r = 0;
        for (int j = 0; j < 32; ++j)
        {
            int8_t it = src1[i * 32 + j];
            int8_t it2 = src2[i * 32 + j];

            r += it > it2 ? it - it2 : it2 - it;
        }

        dst[i] = r;
    }

    return 1;
}


int32_t vec_i32x8n_avg(size_t size, const int32_t *src)
{
    return vec_i32x8n_sum(size, src) / size;
}
double vec_i32x8n_avg_f64(size_t size, const int32_t *src)
{
    return ((double)vec_i32x8n_sum(size, src) / size);
}
int16_t vec_i16x16n_avg(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum(size, src) / size;
}
float vec_i16x16n_avg_f32(size_t size, const int16_t *src)
{
    return (float)((double)vec_i16x16n_sum(size, src) / size);
}
int8_t vec_i8x32n_avg(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum(size, src) / size;
}
float vec_i8x32n_avg_f32(size_t size, const int8_t *src)
{
    return (float)((double)vec_i8x32n_sum(size, src) / size);
}


/* sum */

int32_t vec_i32x8n_sum_i32(size_t size, const int32_t *src)
{
    size_t result = vec_i32x8n_sum(size, src);
    return result > INT32_MAX ? INT32_MAX : result;
}
size_t vec_i32x8n_sum(size_t size, const int32_t *src)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += src[i];
    }

    return result;
}

int16_t vec_i16x16n_sum_i16(size_t size, const int16_t *src)
{
    size_t result = vec_i16x16n_sum(size, src);
    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_i16x16n_sum(size_t size, const int16_t *src)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += src[i];
    }

    return result;
}

int8_t vec_i8x32n_sum_i8(size_t size, const int8_t *src)
{
    size_t result = vec_i8x32n_sum(size, src);
    return result > INT8_MAX ? INT32_MAX
        : result < INT8_MIN ? INT32_MIN
        : result;
}
int16_t vec_i8x32n_sum_i16(size_t size, const int8_t *src)
{
    size_t result = vec_i8x32n_sum(size, src);
    return result > INT16_MAX ? INT32_MAX
        : result < INT16_MIN ? INT32_MIN
        : result;
}
int32_t vec_i8x32n_sum_i32(size_t size, const int8_t *src)
{
    size_t result = vec_i8x32n_sum(size, src);
    return result > INT32_MAX ? INT32_MAX
        : result < INT32_MIN ? INT32_MIN
        : result;
}
size_t vec_i8x32n_sum(size_t size, const int8_t *src)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += src[i];
    }

    return result;
}


/* deviation sum of square */

size_t vec_i32x8n_dss_with_avg(size_t size, const int32_t *src, int32_t _avg)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - _avg;
        result += tmp * tmp;
    }

    return result;
}
int64_t vec_i32x8n_dss_with_avg_i64(size_t size, const int32_t *src, int32_t _avg);
size_t vec_i16x16n_dss_with_avg(size_t size, const int16_t *src, int16_t _avg)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - _avg;
        result += tmp * tmp;
    }

    return result;
}
int32_t vec_i16x16n_dss_with_avg_i32(size_t size, const int16_t *src, int16_t _avg);
int64_t vec_i16x16n_dss_with_avg_i64(size_t size, const int16_t *src, int16_t _avg);
uint32_t vec_u16v16n_dss_with_avg_u32(size_t size, const uint16_t *src, uint16_t _avg);
uint64_t vec_u16v16n_dss_with_avg_u64(size_t size, const uint16_t *src, uint16_t _avg);
size_t vec_i8x32n_dss_with_avg(size_t size, const int8_t *src, int8_t _avg)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - _avg;
        result += tmp * tmp;
    }

    return result;
}
int32_t vec_i8x32n_dss_with_avg_i32(size_t size, const int8_t *src, int8_t _avg);
size_t vec_u8v32n_dss_with_avg(size_t size, const uint8_t *src, uint8_t _avg)
;
uint16_t vec_u8v32n_dss_with_avg_u16(size_t size, const uint8_t *src, uint8_t _avg);
uint32_t vec_u8v32n_dss_with_avg_u32(size_t size, const uint8_t *src, uint8_t _avg);
size_t vec_i32x8n_dss(size_t size, const int32_t *src)
{
    size_t _avg = vec_i32x8n_avg(size, src);
    return vec_i32x8n_dss_with_avg(size, src, _avg);
}
int64_t vec_i32x8n_dss_i64(size_t size, const int32_t *src);
size_t vec_i16x16n_dss(size_t size, const int16_t *src)
{
    size_t _avg = vec_i16x16n_avg(size, src);
    return vec_i16x16n_dss_with_avg(size, src, _avg);
}
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
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - predicted[i];
        result += tmp * tmp;
    }

    return result;
}
int64_t vec_i32x8n_rss_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_rss(size_t size, const int16_t *src, const int16_t *predicted)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - predicted[i];
        result += tmp * tmp;
    }

    return result;
}
int32_t vec_i16x16n_rss_i32(size_t size, const int16_t *src, const int16_t *predicted);
int64_t vec_i16x16n_rss_i64(size_t size, const int16_t *src, const int16_t *predicted);
uint32_t vec_u16v16n_rss_u32(size_t size, const uint16_t *src, const uint16_t *predicted);
uint64_t vec_u16v16n_rss_u64(size_t size, const uint16_t *src, const uint16_t *predicted);
size_t vec_i8x32n_rss(size_t size, const uint8_t *src, const uint8_t *predicted)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        size_t tmp = (size_t)src[i] - predicted[i];
        result += tmp * tmp;
    }

    return result;
}
int16_t vec_i8x32n_rss_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_rss_i32(size_t size, const int8_t *src, const int8_t *predicted);
uint16_t vec_u8v32n_rss_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_rss_u32(size_t size, const uint8_t *src, const uint8_t *predicted);

size_t vec_i32x8n_ess(size_t size, const int32_t *src, const int32_t *predicted)
{
    size_t _avg = vec_i32x8n_avg(size, src);
    return vec_i32x8n_dss_with_avg(size, predicted, _avg);
}
int64_t vec_i32x8n_ess_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_ess(size_t size, const int16_t *src, const int16_t *predicted)
{
    size_t _avg = vec_i16x16n_avg(size, src);
    return vec_i16x16n_dss_with_avg(size, predicted, _avg);
}
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


/* rescale */

// 10101010 -> 1100110011001100
int vec_i16x16n_rescale_i32x16n(size_t input_size, const int16_t *src, int32_t *dst)
{
    int i;
    for (i = 0; i < input_size; ++i)
    {
        uint16_t it = src[i];
        uint32_t it2 = 0;

        for (int j = 0; j < 16; ++j)
        {
            uint32_t it3 = (it >> j) & 1;
            it3 = (it3 << 2) - 1;

            it2 |= it3 << (j * 2);
        }

        dst[i] = it2;
    }

    return 1;
}
int vec_i8x32n_rescale_i32x32n(size_t input_size, const int8_t *src, int32_t *dst)
{
    int i;
    for (i = 0; i < input_size; ++i)
    {
        uint8_t it = src[i];
        uint32_t it2 = 0;

        for (int j = 0; j < 8; ++j)
        {
            uint32_t it3 = (it >> j) & 1;
            it3 = (it3 << 4) - 1;

            it2 |= it3 << (j * 4);
        }

        dst[i] = it2;
    }

    return 1;
}
int vec_i8x16n_rescale_i16x16n(size_t input_size, const int8_t *src, int16_t *dst)
{
    int i;
    for (i = 0; i < input_size; ++i)
    {
        uint8_t it = src[i];
        uint16_t it2 = 0;

        for (int j = 0; j < 8; ++j)
        {
            uint16_t it3 = (it >> j) & 1;
            it3 = (it3 << 2) - 1;

            it2 |= it3 << (j * 2);
        }

        dst[i] = it2;
    }

    return 1;
}


// 1000110001001100 -> 10101010  OR
int vec_i32x16n_rescale_i16x16n(size_t input_size, const int32_t *src, int16_t *dst);
int vec_i32x32n_rescale_i8x32n(size_t input_size, const int32_t *src, int8_t *dst);
int vec_i16x32n_rescale_i8x32n(size_t input_size, const int16_t *src, int8_t *dst);

// 10101010 -> 1000100010001000
int vec_i16x16n_sparse_i32x16n(size_t input_size, const int16_t *src, int32_t *dst);
int vec_i8x32n_sparse_i32x32n(size_t input_size, const int8_t *src, int32_t *dst);
int vec_i8x16n_sparse_i16x16n(size_t input_size, const int8_t *src, int16_t *dst);

// 1000110001001100 -> 00100010  AND
int vec_i32x16n_densify_i16x16n(size_t input_size, const int32_t *src, int16_t *dst);
int vec_i32x32n_densify_i8x32n(size_t input_size, const int32_t *src, int8_t *dst);
int vec_i16x32n_densify_i8x32n(size_t input_size, const int16_t *src, int8_t *dst);


/* assignment/*/

void vec_i32x8n_set_seq(size_t size, int32_t *dst, int32_t start, int32_t diff)
{
    int32_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        dst[i] = it;
        it += diff;
    }
}
void vec_i16x16n_set_seq(size_t size, int16_t *dst, int16_t start, int16_t diff)
{
    int16_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        dst[i] = it;
        it += diff;
    }
}
void vec_i8x32n_set_seq(size_t size, int8_t *dst, int8_t start, int8_t diff)
{
    int8_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        dst[i] = it;
        it += diff;
    }
}

