#include <stddef.h>
#include <stdint.h>

#include "../../include/interleave.h"


/* interleaved/packed array/table */

// src[0], _, _, _, _, _, _, _, src[1], _, _, _, _, _, _, _, ...
int  vec_i32x8xn_set_col(size_t input_size, const int32_t *src, int32_t *dst)
{
    int i;
    for (i = 0; i < input_size; ++i)
    {
        dst[i * 8] = src[i];
    }

    return 1;
}
int  vec_i16x16xn_set_col(size_t input_size, const int16_t *src, int16_t *dst);
int  vec_i8x32xn_set_col(size_t input_size, const int8_t *src, int8_t *dst);
int  vec_i32x8xn_set_col_at(size_t input_size, const int32_t *src, int32_t *dst, int index);
int  vec_i16x16xn_set_col_at(size_t input_size, const int16_t *src, int16_t *dst, int index);
int  vec_i8x32xn_set_col_at(size_t input_size, const int16_t *src, int16_t *dst, int index);

int  vec_i32x8xn_interleave8(size_t input_size, int32_t **src, int32_t *dst)
{
    for (int i = 0; i < 8; ++i)
    {
        int32_t *arr = src[i];

        for (int j = 0; j < input_size; ++j)
        {
            dst[j * input_size + i] = arr[j]
        }
    }

    return 1;
}
int  vec_i16x16xn_interleave16(size_t input_size, int16_t **src, int16_t *dst);
int  vec_i8x32xn_interleave32(size_t input_size, int8_t **src, int8_t *dst);

int  vec_i32x8xn_deinterleave8(size_t input_size, const int32_t *src, int32_t **dst);
int  vec_i16x16xn_deinterleave16(size_t input_size, const int16_t *src, int16_t **dst);
int  vec_i8x32xn_deinterleave32(size_t input_size, const int8_t *src, int8_t **dst);

// ..., a[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i], ... ->
// ..., _, a[i],b[i],c[i],d[i],e[i],f[i],g[i], ...
int  vec_i32x8xn_shr1(size_t input_size, const int32_t *src, int32_t *dst);
int  vec_i16x16xn_shr1(size_t input_size, const int16_t *src, int16_t *dst);
int  vec_i8x32xn_shr1(size_t input_size, const int8_t *src, int8_t *dst);

// result(i32x8x8):
//   src[0], src[8], ..., src[48], src[56], 
//   src[1], src[9], ..., src[49], src[57],
//   ...,
//   src[1], src[9], ..., src[54], src[62],
//   src[7], src[15], ..., src[55], src[63],
//   src[64], src[72], ...,
int  vec_i32x8x8xn_get_rotated(size_t input_size, const int32_t *src, int32_t *dst);
int  vec_i16x16x16xn_get_rotated(size_t input_size, const int16_t *src, int16_t *dst);
int  vec_i8x32x32xn_get_rotated(size_t input_size, const int16_t *src, int16_t *dst);
