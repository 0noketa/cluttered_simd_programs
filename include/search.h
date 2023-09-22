
#ifndef _SIMD_TOOLS__SEARCH__H_
#define _SIMD_TOOLS__SEARCH__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


/* min/max */

size_t vec_i32v8n_get_min_index(size_t size, int32_t *src);
size_t vec_i32v8n_get_max_index(size_t size, int32_t *src);
void vec_i32v8n_get_minmax_index(size_t size, int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32v8n_get_min(size_t size, int32_t *src);
int32_t vec_i32v8n_get_max(size_t size, int32_t *src);
void vec_i32v8n_get_minmax(size_t size, int32_t *src, int32_t *out_min, int32_t *out_max);


size_t vec_i16v16n_get_min_index(size_t size, int16_t *src)
;
size_t vec_i16v16n_get_max_index(size_t size, int16_t *src)
;
void vec_i16v16n_get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max)
;
int16_t vec_i16v16n_get_min(size_t size, int16_t *src)
;
int16_t vec_i16v16n_get_max(size_t size, int16_t *src)
;
void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max)
;

size_t vec_i8v32n_get_min_index(size_t size, int8_t *src)
;
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src)
;
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max)
;
int8_t vec_i8v32n_get_min(size_t size, int8_t *src)
;
int8_t vec_i8v32n_get_max(size_t size, int8_t *src)
;
void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max)
;


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

int32_t vec_i32v8n_count_gt(size_t size, int32_t *src, int32_t value);
int16_t vec_i16v16n_count_gt(size_t size, int16_t *src, int16_t value);
int8_t vec_i8v32n_count_gt(size_t size, int8_t *src, int8_t value);
// any
int32_t vec_i32v8n_get_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_index(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_first_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_first_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_first_index(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_last_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_last_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_last_index(size_t size, int8_t *src, int8_t element);



#endif
