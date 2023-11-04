
#ifndef _SIMD_TOOLS__SEARCH__H_
#define _SIMD_TOOLS__SEARCH__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


/* min/max */

extern int32_t vec_i32v8n_get_min_index_i32(size_t size, const int32_t *src)
;
extern size_t vec_i32v8n_get_min_index(size_t size, const int32_t *src)
;
extern int32_t vec_i32v8n_get_max_index_i32(size_t size, const int32_t *src)
;
extern size_t vec_i32v8n_get_max_index(size_t size, const int32_t *src)
;
extern void vec_i32v8n_get_minmax_index_i32(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max)
;
extern void vec_i32v8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max)
;
extern int32_t vec_i32v8n_get_min(size_t size, const int32_t *src)
;
extern int32_t vec_i32v8n_get_max(size_t size, const int32_t *src)
;
extern void vec_i32v8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max)
;


extern int16_t vec_i16v16n_get_min_index_i16(size_t size, const int16_t *src)
;
extern size_t vec_i16v16n_get_min_index(size_t size, const int16_t *src)
;
extern int16_t vec_i16v16n_get_max_index_i16(size_t size, const int16_t *src)
;
extern size_t vec_i16v16n_get_max_index(size_t size, const int16_t *src)
;
extern int16_t vec_i16v16n_get_minmax_index_i16(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
;
extern void vec_i16v16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max)
;
extern int16_t vec_i16v16n_get_min(size_t size, const int16_t *src)
;
extern int16_t vec_i16v16n_get_max(size_t size, const int16_t *src)
;
extern void vec_i16v16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
;

extern int8_t vec_i8v32n_get_min_index_i8(size_t size, const int8_t *src)
;
extern size_t vec_i8v32n_get_min_index(size_t size, const int8_t *src)
;
extern int8_t vec_i8v32n_get_max_index_i8(size_t size, const int8_t *src)
;
extern size_t vec_i8v32n_get_max_index(size_t size, const int8_t *src)
;
extern void vec_i8v32n_get_minmax_index_i8(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
;
extern void vec_i8v32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max)
;
extern int8_t vec_i8v32n_get_min(size_t size, const int8_t *src)
;
extern int8_t vec_i8v32n_get_max(size_t size, const int8_t *src)
;
extern void vec_i8v32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)
;

extern int32_t vec_i32v8n_get_range(size_t size, const int32_t *src);
extern int16_t vec_i16v16n_get_range(size_t size, const int16_t *src);
extern int8_t vec_i8v32n_get_range(size_t size, const int8_t *src);

extern int32_t vec_i32v8n_get_avg(size_t size, const int32_t *src);
extern int16_t vec_i16v16n_get_avg(size_t size, const int16_t *src);
extern int8_t vec_i8v32n_get_avg(size_t size, const int8_t *src);



/* search */

extern int32_t vec_i32v8n_count_i32(size_t size, const int32_t *src, int32_t value)
;
extern size_t vec_i32v8n_count(size_t size, const int32_t *src, int32_t value)
;
extern int16_t vec_i16v16n_count_i16(size_t size, const int16_t *src, int16_t value)
;
extern size_t vec_i16v16n_count(size_t size, const int16_t *src, int16_t value)
;
extern int8_t vec_i8v32n_count_i8(size_t size, const int8_t *src, int8_t value)
;
extern size_t vec_i8v32n_count(size_t size, const int8_t *src, int8_t value)
;

extern int32_t vec_i32v8n_count_gt(size_t size, const int32_t *src, int32_t value);
extern int16_t vec_i16v16n_count_gt(size_t size, const int16_t *src, int16_t value);
extern int8_t vec_i8v32n_count_gt(size_t size, const int8_t *src, int8_t value);
// any
extern int32_t vec_i32v8n_get_index(size_t size, const int32_t *src, int32_t element);
extern int16_t vec_i16v16n_get_index(size_t size, const int16_t *src, int16_t element);
extern int8_t vec_i8v32n_get_index(size_t size, const int8_t *src, int8_t element);

extern int32_t vec_i32v8n_get_first_index(size_t size, const int32_t *src, int32_t element);
extern int16_t vec_i16v16n_get_first_index(size_t size, const int16_t *src, int16_t element);
extern int8_t vec_i8v32n_get_first_index(size_t size, const int8_t *src, int8_t element);

extern int32_t vec_i32v8n_get_last_index(size_t size, const int32_t *src, int32_t element);
extern int16_t vec_i16v16n_get_last_index(size_t size, const int16_t *src, int16_t element);
extern int8_t vec_i8v32n_get_last_index(size_t size, const int8_t *src, int8_t element);



#endif
