#include <stddef.h>
#include <stdint.h>

#include "../include/simd_tools.h"


/* min/max */

size_t vec_i32v8n_get_min_index(size_t size, int32_t *src);
size_t vec_i32v8n_get_max_index(size_t size, int32_t *src);
void vec_i32v8n_get_minmax_index(size_t size, int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32v8n_get_min(size_t size, int32_t *src);
int32_t vec_i32v8n_get_max(size_t size, int32_t *src);
void vec_i32v8n_get_minmax(size_t size, int32_t *src, int32_t *out_min, int32_t *out_max);


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
    for (i = 0; i < size; ++i)
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


/* search */

int32_t vec_i32v8n_count(size_t size, int32_t *src, int32_t element)
{
    int32_t result = 0;

    for (size_t i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result >= INT16_MAX) break;
        }
    }

    return result;
}
int16_t vec_i16v16n_count(size_t size, int16_t *src, int16_t element)
{
    int16_t result = 0;

    for (size_t i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result >= INT16_MAX) break;
        }
    }

    return result;
}
int8_t vec_i8v32n_count(size_t size, int8_t *src, int8_t element)
{
    int8_t result = 0;

    for (size_t i = 0; i < size; ++i)
    {
        if (src[i] == element)
        {
            ++result;
            if (result >= INT8_MAX) break;
        }
    }

    return result;
}


int32_t vec_i32v8n_get_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_index(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_first_index(size_t size, int32_t *src, int32_t element)
{
    size_t size2 = size > INT32_MAX ? INT32_MAX : size;

    for (size_t i = 0; i < size2; ++i)
    {
        if (src[i] == element) return i;
    }

    return size2;
}
int16_t vec_i16v16n_get_first_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_first_index(size_t size, int8_t *src, int8_t element);
