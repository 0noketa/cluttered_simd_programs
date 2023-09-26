#include <stddef.h>
#include <stdint.h>

#include "../../include/simd_tools.h"

#ifdef USE_LUT
#include "./hw_16bit_lut.c"
#endif

/* local */


// -O2 is faster as mmx/sse2 versions on Atom N2700
void  vec_i16v16n_abs(size_t size, int16_t *src)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int16_t it = src[i];

		if (it < 0) src[i] = -it;
	}
}

void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
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

void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int32_t it = src2[i] - src1[i];

		if (it <= INT16_MIN)
		{
			it = INT16_MAX;
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

size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src)
{
    size_t units = size / 4;
    uint32_t *p = (void*)src;
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


size_t vec_i32v8n_get_sum(size_t size, uint32_t *src)
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:r)
    for (i = 0; i < size; ++i)
    {
        r += src[i];
    }

    return r;
}
int16_t vec_i16v16n_get_sum_i16(size_t size, uint16_t *src)
;
size_t vec_i16v16n_get_sum(size_t size, uint16_t *src)
;
int8_t vec_i8v32n_get_sum_i8(size_t size, uint8_t *src)
;
size_t vec_i8v32n_get_sum(size_t size, uint8_t *src)
;

/* rescale */

// 10101010 -> 1100110011001100
int vec_i16v16n_rescale_i32v16n(size_t input_size, int16_t *src, int32_t *dst);
int vec_i8v32n_rescale_i32v32n(size_t input_size, int8_t *src, int32_t *dst);
int vec_i8v16n_rescale_i16v16n(size_t input_size, int8_t *src, int16_t *dst);

// 1000110001001100 -> 10101010  OR
int vec_i32v16n_rescale_i16v16n(size_t input_size, int32_t *src, int16_t *dst);
int vec_i32v32n_rescale_i8v32n(size_t input_size, int32_t *src, int8_t *dst);
int vec_i16v32n_rescale_i8v32n(size_t input_size, int16_t *src, int8_t *dst);

// 10101010 -> 1000100010001000
int vec_i16v16n_sparse_i32v16n(size_t input_size, int16_t *src, int32_t *dst);
int vec_i8v32n_sparse_i32v32n(size_t input_size, int8_t *src, int32_t *dst);
int vec_i8v16n_sparse_i16v16n(size_t input_size, int8_t *src, int16_t *dst);

// 1000110001001100 -> 00100010  AND
int vec_i32v16n_densify_i16v16n(size_t input_size, int32_t *src, int16_t *dst);
int vec_i32v32n_densify_i8v32n(size_t input_size, int32_t *src, int8_t *dst);
int vec_i16v32n_densify_i8v32n(size_t input_size, int16_t *src, int8_t *dst);


/* assignment/*/

void vec_i32v8n_set_seq(size_t size, int32_t *src, int32_t start, int32_t diff)
{
    int32_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        src[i] = it;
        it += diff;
    }
}
void vec_i16v16n_set_seq(size_t size, int16_t *src, int16_t start, int16_t diff)
{
    int16_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        src[i] = it;
        it += diff;
    }
}
void vec_i8v32n_set_seq(size_t size, int8_t *src, int8_t start, int8_t diff)
{
    int8_t it = start;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        src[i] = it;
        it += diff;
    }
}

