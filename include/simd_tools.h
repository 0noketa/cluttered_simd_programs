
#ifndef _SIMD_TOOLS__H_
#define _SIMD_TOOLS__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


/* minmax */

extern size_t vec_i32v8n_get_min_index(size_t size, const int32_t *src);
extern size_t vec_i32v8n_get_max_index(size_t size, const int32_t *src);
extern void vec_i32v8n_get_minmax_index(size_t size, const int32_t *src, size_t *out_min, size_t *out_max);
extern int32_t vec_i32v8n_get_min(size_t size, const int32_t *src);
extern int32_t vec_i32v8n_get_max(size_t size, const int32_t *src);
extern void vec_i32v8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max);

extern size_t vec_i16v16n_get_min_index(size_t size, const int16_t *src);
extern size_t vec_i16v16n_get_max_index(size_t size, const int16_t *src);
extern void vec_i16v16n_get_minmax_index(size_t size, const int16_t *src, size_t *out_min, size_t *out_max);
extern int16_t vec_i16v16n_get_min(size_t size, const int16_t *src);
extern int16_t vec_i16v16n_get_max(size_t size, const int16_t *src);
extern void vec_i16v16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max);

extern size_t vec_i8v32n_get_min_index(size_t size, const int8_t *src);
extern size_t vec_i8v32n_get_max_index(size_t size, const int8_t *src);
extern void vec_i8v32n_get_minmax_index(size_t size, const int8_t *src, size_t *out_min, size_t *out_max);
extern int8_t vec_i8v32n_get_min(size_t size, const int8_t *src);
extern int8_t vec_i8v32n_get_max(size_t size, const int8_t *src);
extern void vec_i8v32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max);

extern int32_t vec_i32v8n_get_range(size_t size, const int32_t *src);
extern int16_t vec_i16v16n_get_range(size_t size, const int16_t *src);
extern int8_t vec_i8v32n_get_range(size_t size, const int8_t *src);

extern int32_t vec_i32v8n_get_avg(size_t size, const int32_t *src);
extern int16_t vec_i16v16n_get_avg(size_t size, const int16_t *src);
extern int8_t vec_i8v32n_get_avg(size_t size, const int8_t *src);


extern void  vec_i16v16n_abs(size_t size, const int16_t *src);

extern void  vec_i16v16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst);
extern void  vec_i16v16n_dist(size_t size, const int16_t *src1, const int16_t *src2, int16_t *dst);

extern void vec_i32v8n_inplace_reverse(size_t size, int32_t *data);
extern void vec_i16v16n_inplace_reverse(size_t size, int16_t *data);
extern void vec_i8v32n_inplace_reverse(size_t size, int8_t *data);

extern void vec_i32v8n_reverse(size_t size, const int32_t *src, int32_t *dst);
extern void vec_i16v16n_reverse(size_t size, const int16_t *src, int16_t *dst);
extern void vec_i8v32n_reverse(size_t size, const int8_t *src, int8_t *dst);

extern void vec_u256n_shl1(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shr1(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shl8(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shr8(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shl32(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shr32(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_shl(size_t size, uint8_t *src, int n, uint8_t *dst);
extern void vec_u256n_shr(size_t size, uint8_t *src, int n, uint8_t *dst);
extern void vec_u256n_rol1(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_ror1(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_rol8(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_ror8(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_rol32(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_ror32(size_t size, uint8_t *src, uint8_t *dst);
extern void vec_u256n_ror(size_t size, uint8_t *src, int n, uint8_t *dst);
extern void vec_u256n_ror(size_t size, uint8_t *src, int n, uint8_t *dst);


/* hamming weight */

extern size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src);
extern size_t vec_u256n_get_hamming_distance(size_t size, uint8_t *src1, uint8_t *src2);
extern size_t vec_i32v8n_get_hamming_distance(size_t size, const int32_t *src1, const int32_t *src2);
extern size_t vec_i16v16n_get_hamming_distance(size_t size, const int16_t *src1, const int16_t *src2);
extern size_t vec_i8v32n_get_hamming_distance(size_t size, const int8_t *src1, const int8_t *src2);
extern size_t vec_i32v8n_get_manhattan_distance(size_t size, const int32_t *src1, const int32_t *src2);
extern size_t vec_i16v16n_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2);
extern size_t vec_i8v32n_get_manhattan_distance(size_t size, const int8_t *src1, const int8_t *src2);
extern int vec_i16x16xn_get_manhattan_distance(size_t size, const int16_t *src1, const int16_t *src2, int32_t *dst);
extern int vec_i8x32xn_get_manhattan_distance(size_t size, const int8_t *src1, const int8_t *src2, int16_t *dst);


/* sum */

extern int32_t vec_i32v8_sum_i32(size_t size, const int32_t *src);
extern uint32_t vec_u32v8_sum_u32(size_t size, const uint32_t *src);
extern int16_t vec_i16v16_sum_i16(size_t size, const int16_t *src);
extern size_t vec_i16v16_sum(size_t size, const int16_t *src);
extern uint16_t vec_u16v16_sum_u16(size_t size, const uint16_t *src);
extern size_t vec_u16v16_sum(size_t size, const uint16_t *src);
extern int8_t vec_i8v32_sum_i8(size_t size, const int8_t *src);
extern size_t vec_i8v32_sum(size_t size, const int8_t *src);
extern uint8_t vec_u8v32_sum_u8(size_t size, const uint8_t *src);
extern size_t vec_u8v32_sum(size_t size, const uint8_t *src);

/* sorted arrays */

extern void  vec_i32v8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end);
// dst should be i32?
extern void  vec_i16v16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end);
// dst should be i16?
extern void  vec_i8v32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end);
extern int  vec_i32v8n_is_sorted_a(size_t size, const int32_t *src);
extern int  vec_i32v8n_is_sorted_d(size_t size, const int32_t *src);
extern int  vec_i16v16n_is_sorted_a(size_t size, const int16_t *src);
extern int  vec_i16v16n_is_sorted_d(size_t size, const int16_t *src);
extern int  vec_i8v32n_is_sorted_a(size_t size, const int8_t *src);
extern int  vec_i8v32n_is_sorted_d(size_t size, const int8_t *src);
// ascendant or descendant
extern int  vec_i32v8n_is_sorted(size_t size, const int32_t *src);
extern int  vec_i16v16n_is_sorted(size_t size, const int16_t *src);
extern int  vec_i8v32n_is_sorted(size_t size, const int8_t *src);

/* assignment */

extern void vec_i32v8n_set_seq(size_t size, const int32_t *src, int32_t start, int32_t diff);
extern void vec_i16v16n_set_seq(size_t size, const int16_t *src, int16_t start, int16_t diff);
extern void vec_i8v32n_set_seq(size_t size, const int8_t *src, int8_t start, int8_t diff);

/* search */

extern int32_t vec_i32v8n_count_i32(size_t size, const int32_t *src, int32_t value);
extern size_t vec_i32v8n_count(size_t size, const int32_t *src, int32_t value);
extern int16_t vec_i16v16n_count_i16(size_t size, const int16_t *src, int16_t value);
extern size_t vec_i16v16n_count(size_t size, const int16_t *src, int16_t value);
extern int8_t vec_i8v32n_count_i8(size_t size, const int8_t *src, int8_t value);
extern size_t vec_i8v32n_count(size_t size, const int8_t *src, int8_t value);

extern int32_t vec_i32v8n_get_index(size_t size, const int32_t *src, int32_t element);
extern int16_t vec_i16v16n_get_index(size_t size, const int16_t *src, int16_t element);
extern int8_t vec_i8v32n_get_index(size_t size, const int8_t *src, int8_t element);

extern int32_t vec_i32v8n_get_last_index(size_t size, const int32_t *src, int32_t element);
extern int16_t vec_i16v16n_get_last_index(size_t size, const int16_t *src, int16_t element);
extern int8_t vec_i8v32n_get_last_index(size_t size, const int8_t *src, int8_t element);

extern int32_t vec_i32v8n_get_nth_index(size_t size, const int32_t *src, int32_t element, size_t n);
extern int16_t vec_i16v16n_get_nth_index(size_t size, const int16_t *src, int16_t element, size_t n);
extern int8_t vec_i8v32n_get_nth_index(size_t size, const int8_t *src, int8_t element, size_t n);

extern int32_t vec_i32v8n_get_nth_by_key32(size_t size, int32_t *keys, int32_t *vals, int32_t key, size_t n);
extern int32_t vec_i32v8n_get_nth_by_key16(size_t size, int16_t *keys, int32_t *vals, int16_t key, size_t n);
extern int32_t vec_i32v8n_get_nth_by_key8(size_t size, int8_t *keys, int32_t *vals, int8_t key, size_t n);
extern int16_t vec_i16v16n_get_nth_by_key16(size_t size, int16_t *keys, int16_t *vals, int16_t key, size_t n);
extern int8_t vec_i8v32n_get_nth_by_key8(size_t size, int8_t *keys, int8_t *vals, int8_t key, size_t n);

extern void  vec_i32v8n_mask_unique(size_t size, const int32_t *src, int32_t *dst);
extern void  vec_i16v16n_mask_unique(size_t size, const int16_t *src, int16_t *dst);
extern void  vec_i8v32n_mask_unique(size_t size, const int8_t *src, int8_t *dst);


/* rescale */

// 10101010 -> 1100110011001100
extern int vec_i16v16n_rescale_i32v16n(size_t input_size, const int16_t *src, int32_t *dst);
extern int vec_i8v32n_rescale_i32v32n(size_t input_size, const int8_t *src, int32_t *dst);
extern int vec_i8v16n_rescale_i16v16n(size_t input_size, const int8_t *src, int16_t *dst);

// 10101010 -> 1000100010001000
extern int vec_i16v16n_sparse_i32v16n(size_t input_size, const int16_t *src, int32_t *dst);
extern int vec_i8v32n_sparse_i32v32n(size_t input_size, const int8_t *src, int32_t *dst);
extern int vec_i8v16n_sparse_i16v16n(size_t input_size, const int8_t *src, int16_t *dst);

// 1000110001001100 -> 10101010  OR
extern int vec_i32v16n_rescale_i16v16n(size_t input_size, const int32_t *src, int16_t *dst);
extern int vec_i32v32n_rescale_i8v32n(size_t input_size, const int32_t *src, int8_t *dst);
extern int vec_i16v32n_rescale_i8v32n(size_t input_size, const int16_t *src, int8_t *dst);

// 1000110001001100 -> 00100010  AND
extern int vec_i32v16n_densify_i16v16n(size_t input_size, const int32_t *src, int16_t *dst);
extern int vec_i32v32n_densify_i8v32n(size_t input_size, const int32_t *src, int8_t *dst);
extern int vec_i16v32n_densify_i8v32n(size_t input_size, const int16_t *src, int8_t *dst);


/* interleaved/packed array/table */

// src[0], _, _, _, _, _, _, _, src[1], _, _, _, _, _, _, _, ...
extern int  vec_i32x8xn_set_col(size_t input_size, const int32_t *src, int32_t *dst);
extern int  vec_i16x16xn_set_col(size_t input_size, const int16_t *src, int16_t *dst);
extern int  vec_i8x32xn_set_col(size_t input_size, const int8_t *src, int8_t *dst);
extern int  vec_i32x8xn_set_col_at(size_t input_size, const int32_t *src, int32_t *dst, int index);
extern int  vec_i16x16xn_set_col_at(size_t input_size, const int16_t *src, int16_t *dst, int index);
extern int  vec_i8x32xn_set_col_at(size_t input_size, const int16_t *src, int16_t *dst, int index);

extern int  vec_i32x8xn_interleave8(size_t input_size, int32_t **src, int32_t *dst);
extern int  vec_i16x16xn_interleave16(size_t input_size, int16_t **src, int16_t *dst);
extern int  vec_i8x32xn_interleave32(size_t input_size, int8_t **src, int8_t *dst);

extern int  vec_i32x8xn_deinterleave8(size_t input_size, const int32_t *src, int32_t **dst);
extern int  vec_i16x16xn_deinterleave16(size_t input_size, const int16_t *src, int16_t **dst);
extern int  vec_i8x32xn_deinterleave32(size_t input_size, const int8_t *src, int8_t **dst);

// ..., a[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i], ... ->
// ..., _, a[i],b[i],c[i],d[i],e[i],f[i],g[i], ...
extern int  vec_i32x8xn_shr1(size_t input_size, const int32_t *src, int32_t *dst);
extern int  vec_i16x16xn_shr1(size_t input_size, const int16_t *src, int16_t *dst);
extern int  vec_i8x32xn_shr1(size_t input_size, const int8_t *src, int8_t *dst);

// result(i32x8x8):
//   src[0], src[8], ..., src[48], src[56], 
//   src[1], src[9], ..., src[49], src[57],
//   ...,
//   src[1], src[9], ..., src[54], src[62],
//   src[7], src[15], ..., src[55], src[63],
//   src[64], src[72], ...,
extern int  vec_i32x8x8xn_get_rotated(size_t input_size, const int32_t *src, int32_t *dst);
extern int  vec_i16x16x16xn_get_rotated(size_t input_size, const int16_t *src, int16_t *dst);
extern int  vec_i8x32x32xn_get_rotated(size_t input_size, const int16_t *src, int16_t *dst);


#endif
