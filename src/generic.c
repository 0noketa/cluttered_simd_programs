#include <stddef.h>
#include <stdint.h>

#include "../include/simd_tools.h"


/* local */



/* minmax */

size_t vec_i16v16n_get_min_index(size_t size, int16_t *src)
{
    int16_t current = INT16_MAX;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it < current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}

size_t vec_i16v16n_get_max_index(size_t size, int16_t *src)
{
    int16_t current = 0;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it > current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}

void vec_i16v16n_get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
{
    int16_t current_min = INT16_MAX;
    int16_t current_max = 0;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int16_t it = src[i];

        if (it < current_min)
        {
            current_min = it;
            current_min_index = i;
        } else if (it > current_max)
        {
            current_max = it;
            current_max_index = i;
        }

    }

    *out_min = current_min_index;
    *out_max = current_max_index;
}


int16_t vec_i16v16n_get_min(size_t size, int16_t *src)
{
    return src[get_min_index(size, src)];
}
int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
{
    return src[get_max_index(size, src)];
}
void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t min_idx, max_idx;
    get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}


size_t vec_i8v32n_get_min_index(size_t size, int8_t *src)
{
    int8_t current = INT8_MAX;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it < current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src)
{
    int8_t current = INT8_MIN;
    size_t current_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it > current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max)
{
    int8_t current_min = INT8_MAX;
    int8_t current_max = INT8_MIN;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int8_t it = src[i];

        if (it < current_min)
        {
            current_min = it;
            current_min_index = i;
        } else if (it > current_max)
        {
            current_max = it;
            current_max_index = i;
        }

    }

    *out_min = current_min_index;
    *out_max = current_max_index;
}


int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
{
    return src[vec_i8v32n_get_min_index(size, src)];
}
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
{
    return src[vec_i8v32n_get_max_index(size, src)];
}
void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t min_idx, max_idx;
    vec_i8v32n_get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}


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


void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst)
{
	size_t i = 0;
	size_t j = size - 1;
	
	while (i < j)
	{
		dst[i] = src[j];
		dst[j] = src[i];

		++i;
		--j;
	}
}
void vec_i16v16n_get_reversed(size_t size, int16_t *src, int16_t *dst)
{
	size_t i = 0;
	size_t j = size - 1;
	
	while (i < j)
	{
		dst[i] = src[j];
		dst[j] = src[i];

		++i;
		--j;
	}
}
void vec_i8v32n_get_reversed(size_t size, int8_t *src, int8_t *dst)
{
	size_t i = 0;
	size_t j = size - 1;
	
	while (i < j)
	{
		dst[i] = src[j];
		dst[j] = src[i];

		++i;
		--j;
	}
}




/* shift */

void vec_u256n_shl1(size_t size, uint8_t *src, uint8_t *dst)
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
void vec_u256n_shl8(size_t size, uint8_t *src, uint8_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size - 1; ++i)
    {
        dst[i] = src[i + 1];
    }

    dst[size - 1] = 0;
}
void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    int32_t *p = (int32_t*)src;
    int32_t *q = (int32_t*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units - 1; ++i)
    {
        q[i] = p[i + 1];
    }

    q[units - 1] = 0;
}
void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size - 1; ++i)
    {
        dst[i + 1] = src[i];
    }

    dst[0] = 0;
}
void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst)
{
    size_t units = size / 4;
    int32_t *p = (int32_t*)src;
    int32_t *q = (int32_t*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units - 1; ++i)
    {
        q[i + 1] = p[i];
    }

    q[0] = 0;
}
void vec_u256n_rol1(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_rol8(size, src, dst);

    dst[size - 1] = src[0] >> 7;
}
void vec_u256n_rol8(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shl8(size, src, dst);

    dst[size - 1] = src[0];
}
void vec_u256n_rol32(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shl32(size, src, dst);

    size_t units = size / 4;
    int32_t *p = (int32_t*)src;
    int32_t *q = (int32_t*)dst;

    q[units - 1] = p[0];
}
void vec_u256n_ror8(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shr8(size, src, dst);

    dst[0] = src[size - 1];
}
void vec_u256n_ror32(size_t size, uint8_t *src, uint8_t *dst)
{
    vec_u256n_shr32(size, src, dst);

    size_t units = size / 4;
    int32_t *p = (int32_t*)src;
    int32_t *q = (int32_t*)dst;

    q[0] = p[units - 1];
}


/* humming weight */

size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src)
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


/* sorted arrays */

void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
{
	int_fast16_t start = 0;
	int_fast16_t end = 0;

    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
	{
		int_fast16_t it = src[i];

		start += it < element;
		end += it > element;
	}
	
	*out_start = start;
	*out_end = size - end;
}
int  vec_i16v16n_is_sorted_a(size_t size, int16_t *src)
{
    int_fast16_t it = src[0];

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];
        if (it > it2) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i16v16n_is_sorted_d(size_t size, int16_t *src)
{
    int_fast16_t it = src[0];

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];
        if (it < it2) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i16v16n_is_sorted(size_t size, int16_t *src)
{
    int_fast16_t it = src[0];
    int ascendant = 1;
    int descendant = 1;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 1; i < size; ++i)
    {
        int_fast16_t it2 = src[i];

        ascendant &= it <= it2;
        descendant &= it >= it2;
        if (!ascendant && !descendant) return 0;

        it = it2;
    }

    return 1;
}
int  vec_i8v32n_is_sorted_a(size_t size, int8_t *src)
{
    int_fast8_t it = src[0];

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 1; i < size; ++i)
    {
        int_fast8_t it2 = src[i];
        if (it > it2) return 0;

        it = it2;
    }

    return 1;
}

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


int32_t vec_i32v8n_count(size_t size, int32_t *src, int32_t element)
{
    int32_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result < INT16_MAX) break;
        }
    }

    return result;
}
int16_t vec_i16v16n_count(size_t size, int16_t *src, int16_t element)
{
    int16_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result < INT16_MAX) break;
        }
    }

    return result;
}
int8_t vec_i8v32n_count(size_t size, int8_t *src, int8_t element)
{
    int8_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result < INT8_MAX) break;
        }
    }

    return result;
}

int  vec_i16v16n_is_sortable(size_t size, int16_t *src)
{
	return size <= INT16_MAX;
}

void  vec_i16v16n_sort(size_t size, int16_t *src, int16_t *dst)
{
    int i;
    #pragma omp parallel for num_threads(4)
	for (i = 0; i < size; ++i)
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


int vec_i32v8n_find_overlap(size_t size, int32_t *a, size_t *out_index)
{
    for (size_t i = 0; i < size; ++i)
    {
        int32_t x = a[i];

        for (size_t j = 0; j < size; ++j)
        {
            if (i != j && a[j] == x)
            {
                if (out_index) *out_index = i;
                return 1;
            }
        }
    }

    return 0;
}

// broken. pivot points counted number again.
size_t vec_i32v8n_count_overlap(int32_t *a, size_t size)
{
    size_t r = 0;

    for (size_t i = 0; i < size; ++i)
    {
        int32_t x = a[i];

        for (size_t j = 0; j < size; ++j)
        {
            if (i != j && a[j] == x) ++r;
        }
    }

    return r;
}
