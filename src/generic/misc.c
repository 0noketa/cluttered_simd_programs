#include <stddef.h>
#include <stdint.h>

#include "../include/simd_tools.h"


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


/* humming weight */

size_t bits256n_get_humming_weight(size_t size, uint8_t *src);
{
    size_t r = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        uint8_t it = src[i];

        for (uint8_t n = 1; n; n <<= 1)
        {
            r += !!(it & n);
        }
    }

    return r;
}






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

