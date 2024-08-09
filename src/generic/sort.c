#include <stddef.h>
#include <stdint.h>

#include "../../include/sort.h"


/* reverse */

void vec_i32x8n_inplace_reverse(size_t size, int32_t *data)
{
    size_t units = size / 2;

	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < units; ++i)
	{
        int_fast32_t it = data[i];
		data[i] = data[size - 1 - i];
		data[size - 1 - i] = it;
	}
}
void vec_i16x16n_inplace_reverse(size_t size, int16_t *data)
{
    size_t units = size / 2;

	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < units; ++i)
	{
        int_fast16_t it = data[i];
		data[i] = data[size - 1 - i];
		data[size - 1 - i] = it;
	}
}
void vec_i8x32n_inplace_reverse(size_t size, int8_t *data)
{
    size_t units = size / 2;

	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < units; ++i)
	{
        int_fast8_t it = data[i];
		data[i] = data[size - 1 - i];
		data[size - 1 - i] = it;
	}
}

void vec_i32x8n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		dst[size - 1 - i] = src[i];
	}
}
void vec_i16x16n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		dst[size - 1 - i] = src[i];
	}
}
void vec_i8x32n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		dst[size - 1 - i] = src[i];
	}
}
void bits256n_reverse(size_t size, const uint8_t *src, uint8_t *dst)
{
	int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
        uint_fast8_t it = src[i];
        uint_fast8_t it2 = 0;
        for (int j = 0; j < 8; ++j)
        {
            it2 |= (it >> j) & 1;
            it2 <<= 1;
        }
		dst[size - 1 - i] = it2;
	}
}


/* shift */

void bits256n_shl1(size_t size, const uint8_t *src, uint8_t *dst)
{

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size - 1; ++i)
    {
        uint8_t it = src[i] << 1;
        it |= src[i + 1] >> 7;

        dst[i] = it;
    }

    dst[size - 1] = src[size - 1] << 1;
}
void bits256n_shl8(size_t size, const uint8_t *src, uint8_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size - 1; ++i)
    {
        dst[i] = src[i + 1];
    }

    dst[size - 1] = 0;
}
void bits256n_shl32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const int32_t *p = (const int32_t*)src;
    int32_t *q = (int32_t*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units - 1; ++i)
    {
        q[i] = p[i + 1];
    }

    q[units - 1] = 0;
}
void bits256n_shr8(size_t size, const uint8_t *src, uint8_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size - 1; ++i)
    {
        dst[i + 1] = src[i];
    }

    dst[0] = 0;
}
void bits256n_shr32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    const int32_t *p = (const int32_t*)src;
    int32_t *q = (int32_t*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units - 1; ++i)
    {
        q[i + 1] = p[i];
    }

    q[0] = 0;
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

void bits256n_ror1(size_t size, const uint8_t *src, uint8_t *dst)
;
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

void bits256n_rol(size_t size, const uint8_t *src, int n, uint8_t *dst)
;
void bits256n_ror(size_t size, const uint8_t *src, int n, uint8_t *dst)
;


/* ascendant/descendant */

void  vec_i32x8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end)
{
	int_fast32_t start = 0;
	int_fast32_t end = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:start, end)
	for (i = 0; i < size; ++i)
	{
		int_fast32_t it = src[i];

		start += it < element;
		end += it > element;
	}

	*out_start = start;
	*out_end = size - end;
}
void  vec_i16x16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	int_fast16_t start = 0;
	int_fast16_t end = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:start, end)
	for (i = 0; i < size; ++i)
	{
		int_fast16_t it = src[i];

		start += it < element;
		end += it > element;
	}

	*out_start = start;
	*out_end = size - end;
}
void  vec_i8x32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end)
{
	int_fast8_t start = 0;
	int_fast8_t end = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:start, end)
	for (i = 0; i < size; ++i)
	{
		int_fast8_t it = src[i];

		start += it < element;
		end += it > element;
	}

	*out_start = start;
	*out_end = size - end;
}


int  vec_i32x8n_is_sorted_a(size_t size, const int32_t *src)
;
int  vec_i32x8n_is_sorted_d(size_t size, const int32_t *src)
;
int  vec_i32x8n_is_sorted(size_t size, const int32_t *src)
;

int  vec_i16x16n_is_sorted_a(size_t size, const int16_t *src)
{
    int_fast16_t it = src[0];

    for (int i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];
        if (it > it2) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i16x16n_is_sorted_d(size_t size, const int16_t *src)
{
    int_fast16_t it = src[0];

    for (int i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];
        if (it < it2) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i16x16n_is_sorted(size_t size, const int16_t *src)
{
    int_fast16_t it = src[0];
    int ascendant = 1;
    int descendant = 1;

    for (int i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];

        ascendant &= it <= it2;
        descendant &= it >= it2;
        if (!ascendant && !descendant) return 0;

        it = it2;
    }

    return 1;
}

int  vec_i8x32n_is_sorted_a(size_t size, const int8_t *src)
{
    int_fast8_t it = src[0];

    for (int i = 1; i < size; ++i)
    {
        int_fast8_t it2 = src[i];
        if (it > it2) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i8x32n_is_sorted_d(size_t size, const int8_t *src)
;
int  vec_i8x32n_is_sorted(size_t size, const int8_t *src)
;


int  vec_i16x16n_is_sortable(size_t size, const int16_t *src)
{
	return size <= INT16_MAX;
}
void  vec_i16x16n_sort(size_t size, const int16_t *src, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int_fast16_t it = src[i];
		int16_t start, end;

		vec_i16x16n_get_sorted_index(size, src, it, &start, &end);

		for (size_t j = start; j < end; ++j)
		{
			dst[j] = it;
		}
	}
}
