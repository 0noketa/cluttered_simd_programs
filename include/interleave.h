
#ifndef _SIMD_TOOLS__INTERLEAVE__H_
#define _SIMD_TOOLS__INTERLEAVE__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


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
