#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

#include "../../include/search.h"

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


/* search */

int32_t vec_i32v8n_count_i32(size_t size, int32_t *src, int32_t value)
;
size_t vec_i32v8n_count(size_t size, int32_t *src, int32_t value)
;
int16_t vec_i16v16n_count_i16(size_t size, int16_t *src, int16_t value)
;
size_t vec_i16v16n_count(size_t size, int16_t *src, int16_t value)
;
int8_t vec_i8v32n_count_i8(size_t size, int8_t *src, int8_t value)
;
size_t vec_i8v32n_count(size_t size, int8_t *src, int8_t value)
;
