
#ifndef _SIMD_TOOLS__H_
#define _SIMD_TOOLS__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


/* minmax */

size_t vec_i32v8n_get_min_index(size_t size, int32_t *src);
size_t vec_i32v8n_get_max_index(size_t size, int32_t *src);
void vec_i32v8n_get_minmax_index(size_t size, int32_t *src, size_t *out_min, size_t *out_max);
int32_t vec_i32v8n_get_min(size_t size, int32_t *src);
int32_t vec_i32v8n_get_max(size_t size, int32_t *src);
void vec_i32v8n_get_minmax(size_t size, int32_t *src, int32_t *out_min, int32_t *out_max);

size_t vec_i16v16n_get_min_index(size_t size, int16_t *src);
size_t vec_i16v16n_get_max_index(size_t size, int16_t *src);
void vec_i16v16n_get_minmax_index(size_t size, int16_t *src, size_t *out_min, size_t *out_max);
int16_t vec_i16v16n_get_min(size_t size, int16_t *src);
int16_t vec_i16v16n_get_max(size_t size, int16_t *src);
void vec_i16v16n_get_minmax(size_t size, int16_t *src, int16_t *out_min, int16_t *out_max);

size_t vec_i8v32n_get_min_index(size_t size, int8_t *src);
size_t vec_i8v32n_get_max_index(size_t size, int8_t *src);
void vec_i8v32n_get_minmax_index(size_t size, int8_t *src, size_t *out_min, size_t *out_max);
int8_t vec_i8v32n_get_min(size_t size, int8_t *src);
int8_t vec_i8v32n_get_max(size_t size, int8_t *src);
void vec_i8v32n_get_minmax(size_t size, int8_t *src, int8_t *out_min, int8_t *out_max);


void  vec_i16v16n_abs(size_t size, int16_t *src);

void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst);
void  vec_i16v16n_dist(size_t size, int16_t *src1, int16_t *src2, int16_t *dst);

void vec_i32v8n_reverse(size_t size, int32_t *dat);

void vec_i32v8n_get_reversed(size_t size, int32_t *src, int32_t *dst);
void vec_i16v16n_get_reversed(size_t size, int16_t *src, int16_t *dst);
void vec_i8v32n_get_reversed(size_t size, int8_t *src, int8_t *dst);

void vec_u256n_shl1(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shr1(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shl8(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_shl(size_t size, uint8_t *src, int n, uint8_t *dst);
void vec_u256n_shr(size_t size, uint8_t *src, int n, uint8_t *dst);
void vec_u256n_rol1(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_ror1(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_rol8(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_ror8(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_rol32(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_ror32(size_t size, uint8_t *src, uint8_t *dst);
void vec_u256n_ror(size_t size, uint8_t *src, int n, uint8_t *dst);
void vec_u256n_ror(size_t size, uint8_t *src, int n, uint8_t *dst);


/* humming weight */

size_t vec_u256n_safe_size_for_gethummingweight();
size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src);
size_t vec_u512n_safe_size_for_gethummingweight();
size_t vec_u512n_get_humming_weight(size_t size, uint8_t *src);


/* sorted arrays */

void  vec_i32v8n_get_sorted_index(size_t size, int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end);
// dst should be i32?
void  vec_i16v16n_get_sorted_index(size_t size, int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end);
// dst should be i16?
void  vec_i8v32n_get_sorted_index(size_t size, int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end);
int  vec_i32v8n_is_sorted_a(size_t size, int32_t *src);
int  vec_i32v8n_is_sorted_d(size_t size, int32_t *src);
int  vec_i16v16n_is_sorted_a(size_t size, int16_t *src);
int  vec_i16v16n_is_sorted_d(size_t size, int16_t *src);
int  vec_i8v32n_is_sorted_a(size_t size, int8_t *src);
int  vec_i8v32n_is_sorted_d(size_t size, int8_t *src);
// ascendant or descendant
int  vec_i32v8n_is_sorted(size_t size, int32_t *src);
int  vec_i16v16n_is_sorted(size_t size, int16_t *src);
int  vec_i8v32n_is_sorted(size_t size, int8_t *src);

/* assignment */

void vec_i32v8n_set_seq(size_t size, int32_t *src, int32_t start, int32_t diff);
void vec_i16v16n_set_seq(size_t size, int16_t *src, int16_t start, int16_t diff);
void vec_i8v32n_set_seq(size_t size, int8_t *src, int8_t start, int8_t diff);

/* search */

int32_t vec_i32v8n_count(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_count(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_count(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_index(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_last_index(size_t size, int32_t *src, int32_t element);
int16_t vec_i16v16n_get_last_index(size_t size, int16_t *src, int16_t element);
int8_t vec_i8v32n_get_last_index(size_t size, int8_t *src, int8_t element);

int32_t vec_i32v8n_get_nth_index(size_t size, int32_t *src, int32_t element, size_t n);
int16_t vec_i16v16n_get_nth_index(size_t size, int16_t *src, int16_t element, size_t n);
int8_t vec_i8v32n_get_nth_index(size_t size, int8_t *src, int8_t element, size_t n);

int32_t vec_i32v8n_get_nth_by_key32(size_t size, int32_t *keys, int32_t *vals, int32_t key, size_t n);
int32_t vec_i32v8n_get_nth_by_key16(size_t size, int16_t *keys, int32_t *vals, int16_t key, size_t n);
int32_t vec_i32v8n_get_nth_by_key8(size_t size, int8_t *keys, int32_t *vals, int8_t key, size_t n);
int16_t vec_i16v16n_get_nth_by_key16(size_t size, int16_t *keys, int16_t *vals, int16_t key, size_t n);
int8_t vec_i8v32n_get_nth_by_key8(size_t size, int8_t *keys, int8_t *vals, int8_t key, size_t n);

void  vec_i32v86n_mask_unique(size_t size, int32_t *src, int32_t *dst);
void  vec_i16v16n_mask_unique(size_t size, int16_t *src, int16_t *dst);
void  vec_i8v32n_mask_unique(size_t size, int8_t *src, int8_t *dst);


#endif
