#include <stddef.h>
#include <stdint.h>

#include "../../include/search.h"


/* min/max */

int32_t vec_i32x8n_get_min_index_i32(size_t size, const int32_t *src)
{
    size_t idx = vec_i32x8n_get_min_index(size, src);
    return idx > INT32_MAX ? INT32_MAX : idx;
}
size_t vec_i32x8n_get_min_index(size_t size, const int32_t *src)
{
    int32_t current = INT32_MAX;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int32_t it = src[i];

        if (it < current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
int32_t vec_i32x8n_get_max_index_i32(size_t size, const int32_t *src)
{
    size_t idx = vec_i32x8n_get_max_index(size, src);
    return idx > INT32_MAX ? INT32_MAX : idx;
}
size_t vec_i32x8n_get_max_index(size_t size, const int32_t *src)
{
    int32_t current = 0;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int32_t it = src[i];

        if (it > current)
        {
            current = it;
            current_index = i;
        }
    }

    return current_index;
}
void vec_i32x8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max)
{
    int32_t current_min = INT32_MAX;
    int32_t current_max = 0;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
    for (i = 0; i < size; ++i)
    {
        int32_t it = src[i];

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
int32_t vec_i32x8n_get_min(size_t size, const int32_t *src)
{
    return src[vec_i32x8n_get_min_index(size, src)];
}
int32_t vec_i32x8n_get_max(size_t size, const int32_t *src)
{
    return src[vec_i32x8n_get_max_index(size, src)];
}
void vec_i32x8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max)
{
    size_t min_idx, max_idx;
    vec_i32x8n_get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}


int16_t vec_i16x16n_get_min_index_i16(size_t size, const int16_t *src)
{
    size_t idx = vec_i16x16n_get_min_index(size, src);
    return idx > INT16_MAX ? INT16_MAX : idx;
}
size_t vec_i16x16n_get_min_index(size_t size, const int16_t *src)
{
    int16_t current = INT16_MAX;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
int16_t vec_i16x16n_get_max_index_i16(size_t size, const int16_t *src)
{
    size_t idx = vec_i16x16n_get_max_index(size, src);
    return idx > INT16_MAX ? INT16_MAX : idx;
}
size_t vec_i16x16n_get_max_index(size_t size, const int16_t *src)
{
    int16_t current = 0;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
void vec_i16x16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max)
{
    int16_t current_min = INT16_MAX;
    int16_t current_max = 0;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
int16_t vec_i16x16n_get_min(size_t size, const int16_t *src)
{
    return src[vec_i16x16n_get_min_index(size, src)];
}
int16_t vec_i16x16n_get_max(size_t size, const int16_t *src)
{
    return src[vec_i16x16n_get_max_index(size, src)];
}
void vec_i16x16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
{
    size_t min_idx, max_idx;
    vec_i16x16n_get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}


int8_t vec_i8x32n_get_min_index_i8(size_t size, const int8_t *src)
{
    size_t idx = vec_i8x32n_get_min_index(size, src);
    return idx > INT8_MAX ? INT8_MAX : idx;
}
size_t vec_i8x32n_get_min_index(size_t size, const int8_t *src)
{
    int8_t current = INT8_MAX;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
int8_t vec_i8x32n_get_max_index_i8(size_t size, const int8_t *src)
{
    size_t idx = vec_i8x32n_get_max_index(size, src);
    return idx > INT8_MAX ? INT8_MAX : idx;
}
size_t vec_i8x32n_get_max_index(size_t size, const int8_t *src)
{
    int8_t current = INT8_MIN;
    size_t current_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
void vec_i8x32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max)
{
    int8_t current_min = INT8_MAX;
    int8_t current_max = INT8_MIN;
    size_t current_min_index = 0;
    size_t current_max_index = 0;

    int i;
    // #pragma omp parallel for num_threads(4)
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
int8_t vec_i8x32n_get_min(size_t size, const int8_t *src)
{
    return src[vec_i8x32n_get_min_index(size, src)];
}
int8_t vec_i8x32n_get_max(size_t size, const int8_t *src)
{
    return src[vec_i8x32n_get_max_index(size, src)];
}
void vec_i8x32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
{
    size_t min_idx, max_idx;
    vec_i8x32n_get_minmax_index(size, src, &min_idx, &max_idx);

    *out_min = src[min_idx];
    *out_max = src[max_idx];
}

int32_t vec_i32x8n_get_range(size_t size, const int32_t *src)
{
    int32_t min_val, max_val;
    vec_i32x8n_get_minmax(size, src, &min_val, &max_val);

    int32_t r = max_val - min_val;
    if (r < max_val) r = INT32_MAX;

    return r;
}
int16_t vec_i16x16n_get_range(size_t size, const int16_t *src)
{
    int16_t min_val, max_val;
    vec_i16x16n_get_minmax(size, src, &min_val, &max_val);

    int16_t r = max_val - min_val;
    if (r < max_val) r = INT16_MAX;

    return r;
}
int8_t vec_i8x32n_get_range(size_t size, const int8_t *src)
{
    int8_t min_val, max_val;
    vec_i8x32n_get_minmax(size, src, &min_val, &max_val);

    int8_t r = max_val - min_val;
    if (r < max_val) r = INT8_MAX;

    return r;
}


/* search */

int32_t vec_i32x8n_count_i32(size_t size, const int32_t *src, int32_t value)
{
    size_t result = vec_i32x8n_count(size, src, value);
    return result > INT32_MAX ? INT32_MAX : result;
}
size_t vec_i32x8n_count(size_t size, const int32_t *src, int32_t value)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += (src[i] == value);
    }

    return result;
}

int16_t vec_i16x16n_count_i16(size_t size, const int16_t *src, int16_t value)
{
    size_t result = vec_i16x16n_count(size, src, value);
    return result > INT16_MAX ? INT16_MAX : result;
}
size_t vec_i16x16n_count(size_t size, const int16_t *src, int16_t value)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += (src[i] == value);
    }

    return result;
}

int8_t vec_i8x32n_count_i8(size_t size, const int8_t *src, int8_t value)
{
    size_t result = vec_i8x32n_count(size, src, value);
    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_i8x32n_count(size_t size, const int8_t *src, int8_t value)
{
    size_t result = 0;

    int i;
    #pragma omp parallel for num_threads(4) reduction(+:result)
    for (i = 0; i < size; ++i)
    {
        result += (src[i] == value);
    }

    return result;
}


int32_t vec_i32x8n_get_index(size_t size, const int32_t *src, int32_t element);
int16_t vec_i16x16n_get_index(size_t size, const int16_t *src, int16_t element);
int8_t vec_i8x32n_get_index(size_t size, const int8_t *src, int8_t element);

int32_t vec_i32x8n_get_first_index(size_t size, const int32_t *src, int32_t element)
{
    size_t size2 = size > INT32_MAX ? INT32_MAX : size;

    for (size_t i = 0; i < size2; ++i)
    {
        if (src[i] == element) return i;
    }

    return size2;
}
int16_t vec_i16x16n_get_first_index(size_t size, const int16_t *src, int16_t element);
int8_t vec_i8x32n_get_first_index(size_t size, const int8_t *src, int8_t element);


/* hisotgram */

void vec_i8x32n_get_histogram_i8x4(size_t size, const int8_t *src, int8_t _min, int8_t _max, uint8_t *out_bins)
{
    int8_t mins0[4];
    int8_t maxs0[4];
    {
        double w = (_max - _min + 1) / 4;
        for (int i = 0; i < 4; ++i) mins0[i] = _min + w * i;
        for (int i = 0; i < 3; ++i) maxs0[i] = mins0[i + 1] - 1;
        maxs0[3] = _max;
    }

    for (int i = 0; i < 4; ++i) out_bins[i] = 0;

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (mins0[j] <= src[i] && src[i] <= maxs0[j]) out_bins[j]++;
        }
    }
}
void vec_i8x32n_get_histogram_u8x8(size_t size, const int8_t *src, int8_t _min, int8_t _max, uint8_t *out_bins)
{
    int8_t mins0[8];
    int8_t maxs0[8];
    {
        double w = (_max - _min + 1) / 8;
        for (int i = 0; i < 8; ++i) mins0[i] = _min + w * i;
        for (int i = 0; i < 7; ++i) maxs0[i] = mins0[i + 1] - 1;
        maxs0[7] = _max;
    }

    for (int i = 0; i < 8; ++i) out_bins[i] = 0;

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if (mins0[j] <= src[i] && src[i] <= maxs0[j]) out_bins[j]++;
        }
    }
}
void vec_i8x32n_get_histogram_u16x4(size_t size, const int8_t *src, int8_t _min, int8_t _max, uint16_t *out_bins);
void vec_i8x32n_get_histogram_u16x8(size_t size, const int8_t *src, int8_t _min, int8_t _max, uint16_t *out_bins);
void vec_i8x32n_get_histogram_u32x4(size_t size, const int8_t *src, int8_t _min, int8_t _max, uint32_t *out_bins);
