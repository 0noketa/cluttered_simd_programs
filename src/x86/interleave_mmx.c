#include <stddef.h>
#include <stdint.h>
#include <mmintrin.h>

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif

#include "../../include/interleave.h"


/* interleaved/packed array/table */

// src[0], _, _, _, _, _, _, _, src[1], _, _, _, _, _, _, _, ...
int  vec_i32x8xn_set_col(size_t input_size, int32_t *src, int32_t *dst)
{
    size_t units = input_size / 2;
    __m64 *p = (__m64*)src;
    __m64 *q = (__m64*)dst;
    __m64 mask_lo = _mm_set_pi32(0, 0xFFFFFFFF);
    __m64 mask_hi = _mm_set_pi32(0xFFFFFFFF, 0);

    for (int i = 0; i < units; ++i)
    {
        __m64 it0 = p[i];
        __m64 it1 = it0;
        __m64 it2 = q[i * 8];
        __m64 it3 = q[i * 8 + 4];

        it0 = _mm_and_si64(it0, mask_lo);
        it2 = _mm_and_si64(it2, mask_hi);
        it2 = _mm_or_si64(it0, it2);

        it1 = _mm_srli_si64(it1, 32);
        it3 = _mm_and_si64(it3, mask_hi);
        it3 = _mm_or_si64(it1, it3);

        q[i * 8] = it2;
        q[i * 8 + 4] = it3;
    }

    ANY_EMMS();
    return 1;
}
static int vec_i32x8xn_set_col2_(size_t input_size, int32_t *src1, int32_t *src2, int32_t *dst)
{
    size_t units = input_size / 2;
    __m64 *p1 = (__m64*)src1;
    __m64 *p2 = (__m64*)src2;
    __m64 *q = (__m64*)dst;
    __m64 mask_lo = _mm_set_pi32(0, 0xFFFFFFFF);
    __m64 mask_hi = _mm_set_pi32(0xFFFFFFFF, 0);

    for (int i = 0; i < units; ++i)
    {
        __m64 it0 = p1[i];
        __m64 it1 = it0;
        __m64 it2 = p2[i];
        __m64 it3 = it2;

        it0 = _mm_and_si64(it0, mask_lo);
        it2 = _mm_slli_si64(it2, 32);
        it2 = _mm_or_si64(it0, it2);

        it1 = _mm_srli_si64(it1, 32);
        it3 = _mm_and_si64(it3, mask_hi);
        it3 = _mm_or_si64(it1, it3);

        q[i * 8] = it2;
        q[i * 8 + 4] = it3;
    }

    ANY_EMMS();
    return 1;
}
int  vec_i16x16xn_set_col(size_t input_size, int16_t *src, int16_t *dst);
int  vec_i8x32xn_set_col(size_t input_size, int8_t *src, int8_t *dst);
int  vec_i32x8xn_set_col_at(size_t input_size, int32_t *src, int32_t *dst, int index);
int  vec_i16x16xn_set_col_at(size_t input_size, int16_t *src, int16_t *dst, int index);
int  vec_i8x32xn_set_col_at(size_t input_size, int16_t *src, int16_t *dst, int index);

int  vec_i32x8xn_interleave8(size_t input_size, int32_t **src, int32_t *dst)
{
    size_t units = input_size / 2;
    __m64 *q = (__m64*)dst;
    __m64 mask_lo = _mm_set_pi32(0, 0xFFFFFFFF);
    __m64 mask_hi = _mm_set_pi32(0xFFFFFFFF, 0);

    for (int i = 0; i < 4; ++i)
    {
        __m64 *p1 = (__m64*)src[i * 2];
        __m64 *p2 = (__m64*)src[i * 2 + 1];

        for (int j = 0; j < units; ++j)
        {
            __m64 it0 = p1[j];
            __m64 it1 = it0;
            __m64 it2 = p2[j];
            __m64 it3 = it2;

            it0 = _mm_and_si64(it0, mask_lo);
            it2 = _mm_slli_si64(it2, 32);
            it2 = _mm_or_si64(it0, it2);

            it1 = _mm_srli_si64(it1, 32);
            it3 = _mm_and_si64(it3, mask_hi);
            it3 = _mm_or_si64(it1, it3);

            q[j * 8] = it2;
            q[j * 8 + 4] = it3;
        }
    }

    ANY_EMMS();
    return 1;
}
int  vec_i16x16xn_interleave16(size_t input_size, int16_t **src, int16_t *dst);
int  vec_i8x32xn_interleave32(size_t input_size, int8_t **src, int8_t *dst);

int  vec_i32x8xn_deinterleave8(size_t input_size, int32_t *src, int32_t **dst);
int  vec_i16x16xn_deinterleave16(size_t input_size, int16_t *src, int16_t **dst);
int  vec_i8x32xn_deinterleave32(size_t input_size, int8_t *src, int8_t **dst);

// ..., a[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i], ... ->
// ..., _, a[i],b[i],c[i],d[i],e[i],f[i],g[i], ...
int  vec_i32x8xn_shr1(size_t input_size, int32_t *src, int32_t *dst);
int  vec_i16x16xn_shr1(size_t input_size, int16_t *src, int16_t *dst);
int  vec_i8x32xn_shr1(size_t input_size, int8_t *src, int8_t *dst);

// result(i32x8x8):
//   src[0], src[8], ..., src[48], src[56], 
//   src[1], src[9], ..., src[49], src[57],
//   ...,
//   src[1], src[9], ..., src[54], src[62],
//   src[7], src[15], ..., src[55], src[63],
//   src[64], src[72], ...,
int  vec_i32x8x8xn_get_rotated(size_t input_size, int32_t *src, int32_t *dst);
int  vec_i16x16x16xn_get_rotated(size_t input_size, int16_t *src, int16_t *dst);
int  vec_i8x32x32xn_get_rotated(size_t input_size, int16_t *src, int16_t *dst);
