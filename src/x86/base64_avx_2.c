// 181104
// based on generic version
#include <immintrin.h>
#include "base64_avx_2.h"

#define INPUT_BLOCK_SIZE 48


static inline __m256i enc(__m256i src);
static inline __m256i little_to_big_epu32(__m256i src);


void base64_avx_2_enc(
    BASE64_AVX_2_ALIGNED uint8_t *dst,
    const BASE64_AVX_2_ALIGNED uint8_t *src,
    size_t len)
{
    if (src == NULL || dst == NULL)
        return;

    size_t blocks = len / INPUT_BLOCK_SIZE;
    size_t rest = len % INPUT_BLOCK_SIZE;

    const __m128i *p = (void*) src;
    __m256i *q = (void*) dst;

    for (size_t i = 0; i < blocks; ++i)
    {
        __m128i src0_0 = *p++;
        __m128i src0_1 = *p++;
        __m128i src0_2 = *p++;

        // rewrite. or optimizer can do? 
        uint8_t *p0 = src0_0.m128i_u8;
        uint8_t *p1 = src0_1.m128i_u8;
        uint8_t *p2 = src0_2.m128i_u8;
/**/
        __m256i src1_0_0 = _mm256_set_epi32( p1[ 5], p1[ 2], p0[15], p0[12],  p0[ 9], p0[ 6], p0[ 3], p0[ 0] );
        __m256i src1_0_1 = _mm256_set_epi32( p1[ 6], p1[ 3], p1[ 0], p0[13],  p0[10], p0[ 7], p0[ 4], p0[ 1] );
        __m256i src1_0_2 = _mm256_set_epi32( p1[ 7], p1[ 4], p1[ 1], p0[14],  p0[11], p0[ 8], p0[ 5], p0[ 2] );

        __m256i src1_1_0 = _mm256_set_epi32( p2[13], p2[10], p2[ 7], p2[ 4],  p2[ 1], p1[14], p1[11], p1[ 8] );
        __m256i src1_1_1 = _mm256_set_epi32( p2[14], p2[11], p2[ 8], p2[ 5],  p2[ 2], p1[15], p1[12], p1[ 9] );
        __m256i src1_1_2 = _mm256_set_epi32( p2[15], p2[12], p2[ 9], p2[ 6],  p2[ 3], p2[ 0], p1[13], p1[10] );
/**//*
        __m256i src1_0_0 = { .m256i_u32 = { p1[ 5], p1[ 2], p0[15], p0[12],  p0[ 9], p0[ 6], p0[ 3], p0[ 0] }};
        __m256i src1_0_1 = { .m256i_u32 = { p1[ 6], p1[ 3], p1[ 0], p0[13],  p0[10], p0[ 7], p0[ 4], p0[ 1] }};
        __m256i src1_0_2 = { .m256i_u32 = { p1[ 7], p1[ 4], p1[ 1], p0[14],  p0[11], p0[ 8], p0[ 5], p0[ 2] }};

        __m256i src1_1_0 = { .m256i_u32 = { p2[13], p2[10], p2[ 7], p2[ 4],  p2[ 1], p1[14], p1[11], p1[ 8] }};
        __m256i src1_1_1 = { .m256i_u32 = { p2[14], p2[11], p2[ 8], p2[ 5],  p2[ 2], p1[15], p1[12], p1[ 9] }};
        __m256i src1_1_2 = { .m256i_u32 = { p2[15], p2[12], p2[ 9], p2[ 6],  p2[ 3], p2[ 0], p1[13], p1[10] }};
/**/
        __m256i src2_0_0 = _mm256_slli_epi32(src1_0_0, 16);
        __m256i src2_0_1 = _mm256_slli_epi32(src1_0_1, 8);
        __m256i src2_0_2 = src1_0_2;
        __m256i src2_1_0 = _mm256_slli_epi32(src1_1_0, 16);
        __m256i src2_1_1 = _mm256_slli_epi32(src1_1_1, 8);
        __m256i src2_1_2 = src1_1_2;

        __m256i src2_0 = _mm256_or_si256(src2_0_0, src2_0_1);
        src2_0 = _mm256_or_si256(src2_0, src2_0_2);

        __m256i src2_1 = _mm256_or_si256(src2_1_0, src2_1_1);
        src2_1 = _mm256_or_si256(src2_1, src2_1_2);

        __m256i dst0_0 = enc(src2_0);
        __m256i dst0_1 = enc(src2_1);

        __m256i dst_0 = little_to_big_epu32(dst0_0);
        __m256i dst_1 = little_to_big_epu32(dst0_1);

        *q++ = dst_0;
        *q++ = dst_1;
    }

    if (0 < rest)
    {
        uint8_t *p2 = (void*) p;
        uint8_t *q2 = (void*) q;

        base64_any_enc(q2, p2, rest);
    }
}

static inline __m256i enc(__m256i src)
{
    // dst0 = [int6(>>18), int6(>>12), int6(>>6), int6(>>0)
    __m256i src0_0 = src;
    __m256i src0_1 = src;
    __m256i src0_2 = src;
    __m256i src0_3 = src;

    src0_0 = _mm256_slli_epi32(src0_0, 6);
    src0_1 = _mm256_slli_epi32(src0_1, 4);
    src0_2 = _mm256_slli_epi32(src0_2, 2);
    src0_3 = src0_3;

    static const __m256i mask0 = { .m256i_u32 = {
            0x3F000000, 0x3F000000,  0x3F000000, 0x3F000000,
            0x3F000000, 0x3F000000,  0x3F000000, 0x3F000000}};
    static const __m256i mask1 = { .m256i_u32 = {
            0x003F0000, 0x003F0000,  0x003F0000, 0x003F0000,
            0x003F0000, 0x003F0000,  0x003F0000, 0x003F0000}};
    static const __m256i mask2 = { .m256i_u32 = {
            0x00003F00, 0x00003F00,  0x00003F00, 0x00003F00,
            0x00003F00, 0x00003F00,  0x00003F00, 0x00003F00}};
    static const __m256i mask3 = { .m256i_u32 = {
            0x0000003F, 0x0000003F,  0x0000003F, 0x0000003F,
            0x0000003F, 0x0000003F,  0x0000003F, 0x0000003F}};

    __m256i dst0_0 = _mm256_and_si256(src0_0, mask0);
    __m256i dst0_1 = _mm256_and_si256(src0_1, mask1);
    __m256i dst0_2 = _mm256_and_si256(src0_2, mask2);
    __m256i dst0_3 = _mm256_and_si256(src0_3, mask3);
    dst0_0 = _mm256_or_si256(dst0_0, dst0_1);
    dst0_2 = _mm256_or_si256(dst0_2, dst0_3);
    __m256i dst0 = _mm256_or_si256(dst0_0, dst0_2);


    // tochars

    // 26
    static const __m256i filter_up = { .m256i_u64 = {
            0x1A1A1A1A1A1A1A1A, 0x1A1A1A1A1A1A1A1A,
            0x1A1A1A1A1A1A1A1A, 0x1A1A1A1A1A1A1A1A}};
    // 52
    static const __m256i filter_lo = { .m256i_u64 = {
            0x3434343434343434, 0x3434343434343434,
            0x3434343434343434, 0x3434343434343434}};
    // 62 (-> '+'
    static const __m256i filter_0 = { .m256i_u64 = {
            0x3E3E3E3E3E3E3E3E, 0x3E3E3E3E3E3E3E3E,
            0x3E3E3E3E3E3E3E3E, 0x3E3E3E3E3E3E3E3E}};
    static const __m256i filter_sl = { .m256i_u64 = {
            0x3F3F3F3F3F3F3F3F, 0x3F3F3F3F3F3F3F3F,
            0x3F3F3F3F3F3F3F3F, 0x3F3F3F3F3F3F3F3F}};

    // 'A'
    static const __m256i diff_up = { .m256i_u64 = {
            0x4141414141414141, 0x4141414141414141,
            0x4141414141414141, 0x4141414141414141}};
    // 'a' - 26
    static const __m256i diff_lo = { .m256i_u64 = {
            0x4747474747474747, 0x4747474747474747,
            0x4747474747474747, 0x4747474747474747}};
    // char.code('0' & '+' & '/') < base64.element(*)
    // 52
    static const __m256i diff_0_0 = { .m256i_u64 = {
            0x3434343434343434, 0x3434343434343434,
            0x3434343434343434, 0x3434343434343434}};
    // '0'
    static const __m256i seq_0 = { .m256i_u64 = {
            0x3030303030303030, 0x3030303030303030,
            0x3030303030303030, 0x3030303030303030}};
    // '+'
    static const __m256i seq_pl = { .m256i_u64 = {
            0x2B2B2B2B2B2B2B2B, 0x2B2B2B2B2B2B2B2B,
            0x2B2B2B2B2B2B2B2B, 0x2B2B2B2B2B2B2B2B}};
    // '/'
    static const __m256i seq_sl = { .m256i_u64 = {
            0x2F2F2F2F2F2F2F2F, 0x2F2F2F2F2F2F2F2F,
            0x2F2F2F2F2F2F2F2F, 0x2F2F2F2F2F2F2F2F}};
    static const __m256i seq_ff = { .m256i_u64 = {
            UINT64_MAX, UINT64_MAX,
            UINT64_MAX, UINT64_MAX}};
    
    __m256i mask_up = _mm256_cmpgt_epi8(filter_up, dst0);
    __m256i mask_not_up = _mm256_xor_si256(mask_up, seq_ff);
    __m256i mask_lo = _mm256_cmpgt_epi8(filter_lo, dst0);
    mask_lo = _mm256_and_si256(mask_lo, mask_not_up);
    __m256i mask_not_lo = _mm256_xor_si256(mask_lo, seq_ff);
    __m256i mask_not_a = _mm256_and_si256(mask_not_up, mask_not_lo);
    __m256i mask_0 = _mm256_cmpgt_epi8(filter_0, dst0);
    mask_0 = _mm256_and_si256(mask_0, mask_not_a);
    __m256i mask_pl = _mm256_cmpeq_epi8(dst0, filter_0);
    __m256i mask_sl = _mm256_cmpeq_epi8(dst0, filter_sl);

    // [A-Za-z]
    __m256i dst_up = _mm256_adds_epu8(dst0, diff_up);
    __m256i dst_lo = _mm256_adds_epu8(dst0, diff_lo);
    dst_up = _mm256_and_si256(dst_up, mask_up);
    dst_lo = _mm256_and_si256(dst_lo, mask_lo);
    __m256i dst_a = _mm256_or_si256(dst_up, dst_lo);

    // [0-9\+\/]
    __m256i dst_0_0 = _mm256_subs_epu8(dst0, diff_0_0);
    __m256i dst_0_1 = _mm256_adds_epu8(dst_0_0, seq_0);
    __m256i dst_0 = _mm256_and_si256(dst_0_1, mask_0);
    __m256i dst_pl = _mm256_and_si256(mask_pl, seq_pl);
    __m256i dst_sl = _mm256_and_si256(mask_sl, seq_sl);

    // gather
    __m256i dst_s_0 = _mm256_or_si256(dst_a, dst_0);
    __m256i dst_s_1 = _mm256_or_si256(dst_pl, dst_sl);
    __m256i dst = _mm256_or_si256(dst_s_0, dst_s_1);

    return dst;
}

static inline __m256i little_to_big_epu32(__m256i src)
{
    static const __m256i mask = { .m256i_u32 = {
            0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF}};

    __m256i dst0_0 = _mm256_srli_epi32(src, 24);
    __m256i dst0_1 = _mm256_srli_epi32(src, 16);
    __m256i dst0_2 = _mm256_srli_epi32(src, 8);
    __m256i dst0_3 = src;

    __m256i dst1_0 = dst0_0;
    __m256i dst1_1 = _mm256_and_si256(dst0_1, mask);
    __m256i dst1_2 = _mm256_and_si256(dst0_2, mask);
    __m256i dst1_3 = _mm256_and_si256(dst0_3, mask);

    __m256i dst2_0 = dst1_0;
    __m256i dst2_1 = _mm256_slli_epi32(dst1_1, 8);
    __m256i dst2_2 = _mm256_slli_epi32(dst1_2, 16);
    __m256i dst2_3 = _mm256_slli_epi32(dst1_3, 24);

    __m256i dst_lo = _mm256_or_si256(dst2_0, dst2_1);
    __m256i dst_hi = _mm256_or_si256(dst2_2, dst2_3);

    return _mm256_or_si256(dst_hi, dst_lo);
}

bool base64_avx_2_dec(
    BASE64_AVX_2_ALIGNED uint8_t *dst,
    const BASE64_AVX_2_ALIGNED uint8_t *src,
    size_t len)
{
    return base64_any_dec(dst, src, len);
}

