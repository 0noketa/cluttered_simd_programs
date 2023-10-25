#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

#include "../../include/simd_tools.h"

#ifdef _MSC_VER
#define INIT_I16X4(n) { .n64_i16 = { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) } }
#define INIT_I8X8(n) { .n64_i8 = { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) } }
#else
#define INIT_I16X4(n) { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) }
#define INIT_I8X8(n) { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) }
#endif

/* local */

#if 0
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
#endif


/* humming weight */

#ifdef USE_128BIT_UNITS
static inline uint32x2_t vec_u256n_get_hamming_weight_i(uint32x2_t it, uint32x2_t it2, uint32x2_t rs,  uint32x2_t mask, uint32x2_t mask2, uint32x2_t mask3, uint32x2_t mask4, uint32x2_t mask5)
{
    uint32x2_t tmp = vshr_n_u32(it, 1);
    uint32x2_t tmp2 = vshr_n_u32(it2, 1);
    it = vand_u32(it, mask);
    it2 = vand_u32(it2, mask);
    tmp = vand_u32(tmp, mask);
    tmp2 = vand_u32(tmp2, mask);
    it = vadd_u32(it, tmp);
    it2 = vadd_u32(it2, tmp2);

    tmp = vshr_n_u32(it, 2);
    tmp2 = vshr_n_u32(it2, 2);
    it = vand_u32(it, mask2);
    it2 = vand_u32(it2, mask2);
    tmp = vand_u32(tmp, mask2);
    tmp2 = vand_u32(tmp2, mask2);
    it = vadd_u32(it, tmp);
    it2 = vadd_u32(it2, tmp2);

    tmp = vshr_n_u32(it, 4);
    tmp2 = vshr_n_u32(it2, 4);
    it = vand_u32(it, mask3);
    it2 = vand_u32(it2, mask3);
    tmp = vand_u32(tmp, mask3);
    tmp2 = vand_u32(tmp2, mask3);
    it = vadd_u32(it, tmp);
    it2 = vadd_u32(it2, tmp2);

    tmp = vshr_n_u32(it, 8);
    tmp2 = vshr_n_u32(it2, 8);
    it = vand_u32(it, mask4);
    it2 = vand_u32(it2, mask4);
    tmp = vand_u32(tmp, mask4);
    tmp2 = vand_u32(tmp2, mask4);
    it = vadd_u32(it, tmp);
    it2 = vadd_u32(it2, tmp2);

    tmp = vshr_n_u32(it, 16);
    tmp2 = vshr_n_u32(it2, 16);
    it = vand_u32(it, mask5);
    it2 = vand_u32(it2, mask5);
    tmp = vand_u32(tmp, mask5);
    tmp2 = vand_u32(tmp2, mask5);
    it = vadd_u32(it, tmp);
    it2 = vadd_u32(it2, tmp2);

    rs = vadd_U32(rs, it);
    rs = vadd_u32(rs, it2);

    return rs;
}
#else
static inline uint32x2_t vec_u256n_get_hamming_weight_i(uint32x2_t it, uint32x2_t rs,  uint32x2_t mask, uint32x2_t mask2, uint32x2_t mask3, uint32x2_t mask4, uint32x2_t mask5)
{
    uint32x2_t tmp = vshr_n_u32(it, 1);
    it = vand_u32(it, mask);
    tmp = vand_u32(tmp, mask);
    it = vadd_u32(it, tmp);

    tmp = vshr_n_u32(it, 2);
    it = vand_u32(it, mask2);
    tmp = vand_u32(tmp, mask2);
    it = vadd_u32(it, tmp);

    tmp = vshr_n_u32(it, 4);
    it = vand_u32(it, mask3);
    tmp = vand_u32(tmp, mask3);
    it = vadd_u32(it, tmp);

    tmp = vshr_n_u32(it, 8);
    it = vand_u32(it, mask4);
    tmp = vand_u32(tmp, mask4);
    it = vadd_u32(it, tmp);

    tmp = vshr_n_u32(it, 16);
    it = vand_u32(it, mask5);
    tmp = vand_u32(tmp, mask5);
    it = vadd_u32(it, tmp);

    rs = vadd_u32(rs, it);

    return rs;
}
#endif
size_t vec_u256n_get_humming_weight(size_t size, const uint8_t *src)
#ifdef USE_128BIT_UNITS
{
    size_t units = size / 8 / 2;

    const uint32x2_t *p = (const uint32x2_t*)src;

    uint32x2_t rs;
    rs = vxor_u32(rs, rs);

    const uint32_t mask_value = 0x55555555;
    const uint32_t mask2_value = 0x33333333;
    const uint32_t mask3_value = 0x0F0F0F0F;
    const uint32_t mask4_value = 0x0000FFFF;
    uint32x2_t mask = vld1_dup_u32(&mask_value);
    uint32x2_t mask2 = vld1_dup_u32(&mask2_value);
    uint32x2_t mask3 = vld1_dup_u32(&mask3_value);
    uint32x2_t mask4 = vld1_dup_u32(&mask4_value);

    for (size_t i = 0; i < units; ++i)
    {
        uint32x2_t it = vld1_u32(p + i * 2);
        uint32x2_t it2 = vld1_u32(p + i * 2 + 1);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = vget_lane_u32(rs, 0);
    size_t r1 = vget_lane_ui32(rs, 1);

    size_t r = r0 + r1;

    return r;
}
#else
{
    const uint32x2_t *p = (const uint32x2_t*)src;

    uint32x2_t rs;
    rs = vxor_u32(rs, rs);

     const uint32_t mask_value = 0x55555555;
    const uint32_t mask2_value = 0x33333333;
    const uint32_t mask3_value = 0x0F0F0F0F;
    const uint32_t mask4_value = 0x00FF00FF;
    const uint32_t mask5_value = 0x0000FFFF;
    uint32x2_t mask = vld1_dup_u32(&mask_value);
    uint32x2_t mask2 = vld1_dup_u32(&mask2_value);
    uint32x2_t mask3 = vld1_dup_u32(&mask3_value);
    uint32x2_t mask4 = vld1_dup_u32(&mask4_value);
    uint32x2_t mask5 = vld1_dup_u32(&mask5_value);

    for (size_t i = 0; i < units; ++i)
    {
        uint32x2_t it = p[i];

        rs = vec_u256n_get_hamming_weight_i(it, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = vget_lane_u32(rs, 0);
    size_t r1 = vget_lane_u32(rs, 1);

    size_t r = r0 + r1;

    return r;
}
#endif

size_t vec_u256n_get_hamming_distance(size_t size, const uint8_t *src1, const uint8_t *src2)
#ifdef USE_128BIT_UNITS
{
    size_t units = size / 8 / 2;

    const uint32x2_t *p = (const uint32x2_t*)src1;
    uint32x2_t *q = (uint32x2_t*)src2;

    uint32x2_t rs;
    rs = vxor_u32(rs, rs);

    const uint32_t mask_value = 0x55555555;
    const uint32_t mask2_value = 0x33333333;
    const uint32_t mask3_value = 0x0F0F0F0F;
    const uint32_t mask4_value = 0x0000FFFF;
    uint32x2_t mask = vld1_dup_u32(&mask_value);
    uint32x2_t mask2 = vld1_dup_u32(&mask2_value);
    uint32x2_t mask3 = vld1_dup_u32(&mask3_value);
    uint32x2_t mask4 = vld1_dup_u32(&mask4_value);

    for (size_t i = 0; i < units; ++i)
    {
        uint32x2_t it = vld1_u32(p + i * 2);
        uint32x2_t it2 = vld1_u32(p + i * 2 + 1);
        uint32x2_t it_b = vld1_u32(q + i * 2);
        uint32x2_t it2_b = vld1_u32(q + i * 2 + 1);

        it = vxor_u32(it, it_b);
        it2 = vxor_u32(it2, it2_b);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = vget_lane_u32(rs, 0);
    size_t r1 = vget_lane_ui32(rs, 1);

    size_t r = r0 + r1;

    return r;
}
#else
{
    const uint32x2_t *p = (const uint32x2_t*)src;

    uint32x2_t rs;
    rs = vxor_u32(rs, rs);

     const uint32_t mask_value = 0x55555555;
    const uint32_t mask2_value = 0x33333333;
    const uint32_t mask3_value = 0x0F0F0F0F;
    const uint32_t mask4_value = 0x00FF00FF;
    const uint32_t mask5_value = 0x0000FFFF;
    uint32x2_t mask = vld1_dup_u32(&mask_value);
    uint32x2_t mask2 = vld1_dup_u32(&mask2_value);
    uint32x2_t mask3 = vld1_dup_u32(&mask3_value);
    uint32x2_t mask4 = vld1_dup_u32(&mask4_value);
    uint32x2_t mask5 = vld1_dup_u32(&mask5_value);

    for (size_t i = 0; i < units; ++i)
    {
        uint32x2_t it = vld1_u32(p + i);
        uint32x2_t it_b = vld1_u32(q + i);

        it = vxor_u32(it, it_b);

        rs = vec_u256n_get_hamming_weight_i(it, rs,  mask,mask2,mask3,mask4,mask5);
    }

    size_t r0 = vget_lane_u32(rs, 0);
    size_t r1 = vget_lane_u32(rs, 1);

    size_t r = r0 + r1;

    return r;
}
#endif




/* sum */

int32_t vec_i32v8n_sum_i32(size_t size, const int32_t *src)
{
	size_t units = size / 2 / 4;
	const int32x2_t *p = (const int32x2_t*)src;

    int32x2_t rs;
    rs = vxor_i32(rs, re);

	for (size_t i = 0; i < units; ++i)
	{
        int32x2_t it0 = vld1_i32(p + i * 4);
        int32x2_t it1 = vld1_i32(p + i * 4 + 1);
        int32x2_t it2 = vld1_i32(p + i * 4 + 2);
        int32x2_t it3 = vld1_i32(p + i * 4 + 3);

        it0 = vadd_i32(it0, it1);
        it2 = vadd_i32(it2, it3);
        it0 = vadd_i32(it0, it2);
        rs = vadd_i32(rs, it0);
	}

    int32_t r = vget_lane_i32(rs, 0);
    r += vget_lane_i32(rs, 1);

    return r;
}
uint32_t vec_u32v8n_sum_u32(size_t size, const uint32_t *src)
;

int16_t vec_i16v16n_sum_i16(size_t size, const int16_t *src)
;
size_t vec_i16v16n_sum(size_t size, const int16_t *src)
;
uint16_t vec_u16v16n_sum_u16(size_t size, const uint16_t *src)
;
size_t vec_u16v16n_sum(size_t size, const uint16_t *src)
;

// returns i8x8
static int16x4_t vec_i8v32n_sum_i8x8(size_t size, const int8_t *src)
{
    size_t units = size / 16;
    const int8x8_t *p = (const int8x8_t*)src;

    int8x8_t results, results2;
    results = vxor_i8(results, results);
    results2 = vxor_i8(results2, results2);

    for (int i = 0; i < units; ++i)
    {
        int8x8_t it = p[i * 2];
        int8x8_t it2 = p[i * 2 + 1];

        results = vadd_i8(results, it);
        results2 = vadd_i8(results2, it2);
    }

    return vadd_i8(results, results2);
}

int8_t vec_i8v32n_sum_i8(size_t size, const int8_t *src)
{
    int8x8_t results = vec_i8v32n_sum_u8x8(size, src);
    results = vadd_i8(results, vreinterpret_i8_u64(vshr_n_u64(vreinterpret_u64_i8(results), 16)));
    results = vadd_i8(results, vreinterpret_i8_u64(vshr_n_u64(vreinterpret_u64_i8(results), 32)));
    int8_t result = vget_lane_v8(results, 0);

    return result;
}
size_t vec_i8v32n_sum(size_t size, const int8_t *src)
;

// returns i16x4
static uint16x4_t vec_u8v32n_sum_u16x4(size_t size, const uint8_t *src)
{
    size_t units = size / 16;
    const uint16x4_t *p = (const uint16x4_t*)src;

    const uint16x4_t mask_lower = vld1_dup_u16(0x00FF);
    uint16x4_t results = vld1_dup_u16(0);
    uint16x4_t results2 = vld1_dup_u16(0);

    for (int i = 0; i < units; ++i)
    {
        uint16x4_t it = p[i * 2];
        uint16x4_t it2 = p[i * 2 + 1];

        uint16x4_t it_hi = vshl_n_u16(it, 8);
        uint16x4_t it2_hi = vshl_n_u16(it2, 8);
        uint16x4_t it_lo = vand_u16(it, mask_lower);
        uint16x4_t it2_lo = vand_u16(it2, mask_lower);

        results = vadd_u16(results, it_hi);
        results2 = vadd_u16(results2, it2_hi);
        results = vadd_u16(results, it_lo);
        results2 = vadd_u16(results2, it2_lo);
    }

    return vadd_u16(results, results2);
}
uint8_t vec_u8v32n_sum_u8(size_t size, const uint8_t *src)
{
    uint16x4_t results = vec_u8v32n_sum_u16x4(size, src);
    results = vadd_i16(results, vreinterpret_i16_u64(vshr_n_u64(vreinterpret_u64_i16(results), 16)));
    results = vadd_i16(results, vreinterpret_i16_u64(vshr_n_u64(vreinterpret_u64_i16(results), 32)));
    size_t result = vget_lane_u16(results, 0);

    return result > UINT8_MAX ? UINT8_MAX : result;
}
size_t vec_u8v32n_sum(size_t size, const uint8_t *src)
{
    const size_t unit_size = 0x8000 / (UINT8_MAX + 1) * 4;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        uint32x2_t results = vreinterpret_u32_u16(vec_i8v32n_sum_u16x4(unit_size, src + i * unit_size));
        const uint32x2_t mask_lower = vld1_dup_u32(0x0000FFFF);
        uint32x2_t results2 = _mm_srli_si32(results, 16);
        results = vand_u32(results, mask_lower);
        results = vadd_u32(results, results2);
        results = vadd_u32(results, vreinterpret_u32_u64(vshr_n_u64(vreinterpret_u64_u32(results), 32)));
        size_t result2 = vget_lane_u32(results, 0);

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        uint32x2_t results = vreinterpret_u32_u16(vec_i8v32n_sum_u16x4(size2, src + base));
        const uint32x2_t mask_lower = vld1_dup_u32(0x0000FFFF);
        uint32x2_t results2 = _mm_srli_si32(results, 16);
        results = vand_u32(results, mask_lower);
        results = vadd_u32(results, results2);
        results = vadd_u32(results, vreinterpret_u32_u64(vshr_n_u64(vreinterpret_u64_u32(results), 32)));
        size_t result2 = vget_lane_u32(results, 0);
    }

    return result;
}




/* assignment */

void vec_i32v8n_set_seq(size_t size, const int32_t *src, int32_t start, int32_t diff)
;

