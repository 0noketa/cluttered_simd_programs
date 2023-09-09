// 2023-09-08
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <immintrin.h>


#define base16_avx_enc base16_avx2_enc
#define base16_avx_dec base16_avx2_dec

static const __m256i mask_0f = { .m256i_u64 = {
        0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
        0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F}};
static const __m256i mask_ff = { .m256i_u64 = {
        UINT64_MAX, UINT64_MAX,
        UINT64_MAX, UINT64_MAX}};
static const __m256i mask_u16_lo = { .m256i_u64 = {
        0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF,
        0x00FF00FF00FF00FF, 0x00FF00FF00FF00FF}};
static const __m256i mask_u16_hi = { .m256i_u64 = {
        0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00,
        0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00}};
static const __m256i mask_u32_lo = { .m256i_u64 = {
        0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF,
        0x0000FFFF0000FFFF, 0x0000FFFF0000FFFF}};
static const __m256i mask_u32_hi = { .m256i_u64 = {
        0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000,
        0xFFFF0000FFFF0000, 0xFFFF0000FFFF0000}};
static const __m256i mask_u64_lo = { .m256i_u64 = {
        0x00000000FFFFFFFF, 0x00000000FFFFFFFF,
        0x00000000FFFFFFFF, 0x00000000FFFFFFFF}};
static const __m256i mask_u64_hi = { .m256i_u64 = {
        0xFFFFFFFF00000000, 0xFFFFFFFF00000000,
        0xFFFFFFFF00000000, 0xFFFFFFFF00000000}};
static const __m256i mask_u128_lo = { .m256i_u64 = {
        0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
        0xFFFFFFFFFFFFFFFF, 0x0000000000000000}};
static const __m256i mask_u128_hi = { .m256i_u64 = {
        0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
        0x0000000000000000, 0xFFFFFFFFFFFFFFFF}};

// '0' - 0
static const __m256i diff_0 = { .m256i_u64 = {
        0x3030303030303030, 0x3030303030303030,
        0x3030303030303030, 0x3030303030303030}};
// ('A' or 'a') - 10
static const __m256i diff_a_upper = { .m256i_u64 = {
        0x3737373737373737, 0x3737373737373737,
        0x3737373737373737, 0x3737373737373737}};
static const __m256i diff_a_lower = { .m256i_u64 = {
        0x5757575757575757, 0x5757575757575757,
        0x5757575757575757, 0x5757575757575757}};

// comparables
static const __m256i filter_0 = { .m256i_u64 = {
        0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000}};
static const __m256i filter_9 = { .m256i_u64 = {
        0x0909090909090909, 0x0909090909090909,
        0x0909090909090909, 0x0909090909090909}};
static const __m256i filter_10 = { .m256i_u64 = {
        0x0A0A0A0A0A0A0A0A, 0x0A0A0A0A0A0A0A0A,
        0x0A0A0A0A0A0A0A0A, 0x0A0A0A0A0A0A0A0A}};
static const __m256i filter_16 = { .m256i_u64 = {
        0x1010101010101010, 0x1010101010101010,
        0x1010101010101010, 0x1010101010101010}};
static const __m256i filter_26 = { .m256i_u64 = {
        0x1A1A1A1A1A1A1A1A, 0x1A1A1A1A1A1A1A1A,
        0x1A1A1A1A1A1A1A1A, 0x1A1A1A1A1A1A1A1A}};
static const __m256i filter_6 = { .m256i_u64 = {
        0x0606060606060606, 0x0606060606060606,
        0x0606060606060606, 0x0606060606060606}};

// ('A' or 'a') - 1
static const __m256i filter_lower = { .m256i_u64 = {
        0x6060606060606060, 0x6060606060606060,
        0x6060606060606060, 0x6060606060606060}};
static const __m256i filter_upper = { .m256i_u64 = {
        0x4040404040404040, 0x4040404040404040,
        0x4040404040404040, 0x4040404040404040}};


static inline __m256i tochars(__m256i src, __m256i diff_a);


static inline uint8_t enc_char(uint8_t x, bool upper)
{
    uint8_t a  = (upper ? 'A' : 'a') - 10;
    return x + (x < 10 ? '0' : a);
}
static void base16_any_enc(
    uint8_t *dst,
    const uint8_t *src,
    size_t input_size, bool upper)
{
    if (src == NULL || dst == NULL)
        return;

    for (size_t i = 0; i < input_size; ++i)
    {
        size_t j = i << 1;
        uint8_t n = src[i];
        uint8_t n_hi = n >> 4l;
        uint8_t n_lo = n & 0x0F;

        dst[j] = enc_char(n_hi, upper);
        dst[j + 1] = enc_char(n_lo, upper);
    }
}


void base16_avx_enc(
    uint8_t *dst,
    const uint8_t *src,
    size_t input_size, bool upper)
{
    if (src == NULL || dst == NULL)
        return;

    __m256i diff_a = upper ? diff_a_upper : diff_a_lower;

    size_t units = input_size / sizeof(__m256i);
    size_t rest = input_size % sizeof(__m256i);

    const __m256i *p = (void*) src;
    __m256i *q = (void*) dst;

    for (size_t i = 0; i < units; ++i)
    {
        // 67452301 (ex in v4i8) -> 06040200, 07050301
        __m256i src0 = _mm256_load_si256(p++);
        __m256i dst0_hi = _mm256_srli_epi64(src0, 4); 
        dst0_hi = _mm256_and_si256(dst0_hi, mask_0f);
        __m256i dst0_lo = _mm256_and_si256(src0, mask_0f);

        // 07050301, 06040200 -> 07060302, 05040100
        __m256i dst1_lo = _mm256_unpacklo_epi8(dst0_hi, dst0_lo);
        __m256i dst1_hi = _mm256_unpackhi_epi8(dst0_hi, dst0_lo);

        __m128i x = _mm256_extracti128_si256(dst1_lo, 1);
        __m128i y = _mm256_extracti128_si256(dst1_hi, 0);
        __m256i dst2_lo = _mm256_inserti128_si256(dst1_lo, y, 1);
        __m256i dst2_hi = _mm256_inserti128_si256(dst1_hi, x, 0);

        __m256i dst_hi = tochars(dst2_hi, diff_a);
        __m256i dst_lo = tochars(dst2_lo, diff_a);

        _mm256_store_si256(q++, dst_lo);
        _mm256_store_si256(q++, dst_hi);
    }

    const uint8_t *src_r = (void*) p;
    uint8_t *dst_r = (void*) q;

    base16_any_enc(dst_r, src_r, rest, upper);
}

static inline __m256i tochars(__m256i src, __m256i diff_a)
{ 
    // [int4] -> _0: [int4 + '0'], _a: [int4 + 'A' - 10]
    __m256i dst0_0 = _mm256_adds_epu8(src, diff_0);
    __m256i dst0_a = _mm256_adds_epu8(src, diff_a);
    __m256i mask_0 = _mm256_cmpgt_epi8(filter_10, src);
    __m256i mask_a = _mm256_cmpgt_epi8(src, filter_9);
    __m256i dst_0 = _mm256_and_si256(dst0_0, mask_0);
    __m256i dst_a = _mm256_and_si256(dst0_a, mask_a);

    __m256i dst = _mm256_or_si256(dst_0, dst_a);

    return dst;
}




static inline uint8_t dec_char4(uint8_t x)
{
    uint8_t d = isupper(x) ? 'A' - 10
            : islower(x) ? 'a' - 10
            : isdigit(x) ? '0'
            : 0xFF;

    return d == 0xFF ? d : x - d;
}

static inline bool dec_char(uint8_t *dst, uint8_t hi, uint8_t lo)
{
    uint8_t hi2 = dec_char4(hi);
    uint8_t lo2 = dec_char4(lo);

    if ((hi2 | lo2) == 0xFF)
        return false;
    
    *dst = (hi2 << 4) | lo2;

    return true;
}
static bool base16_any_dec(
    uint8_t *dst,
    const uint8_t *src,
    size_t input_size)
{
    if ((input_size & 1) != 0 || dst == NULL || src == NULL)
        return false;

    size_t len2 = input_size >> 1;

    for (size_t i = 0; i < len2; ++i)
    {
        size_t j = i << 1;
        uint8_t hi = src[j];
        uint8_t lo = src[j + 1];
        uint8_t c;

        if (dec_char(&c, hi, lo))
            dst[i] = c;
        else
            return false;
    }

    return true;
}


int base16_128n_decode(size_t input_size, uint32_t *dst, const uint32_t *src)
{
    size_t units = input_size / sizeof(__m256i);

    const __m256i *p = (void*)src;
    __m128i *q = (void*)dst;

    // for (size_t i = 0; i < units; ++i)
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        // __m256i src0 = *p++;
        __m256i src0 = _mm256_load_si256(p++);

        __m256i dst0;
        {
            __m256i dst0_lo = _mm256_subs_epu8(src0, diff_a_lower);
            __m256i mask_lo = _mm256_cmpgt_epi8(src0, filter_lower);
            dst0_lo = _mm256_and_si256(dst0_lo, mask_lo);

            mask_lo = _mm256_cmpgt_epi8(filter_16, dst0_lo);
            dst0_lo = _mm256_and_si256(dst0_lo, mask_lo);

            dst0 = dst0_lo;
        }

        {
            __m256i dst0_up = _mm256_subs_epi8(src0, diff_a_upper);
            __m256i mask_up = _mm256_cmpgt_epi8(src0, filter_upper);
            dst0_up = _mm256_and_si256(dst0_up, mask_up);

            mask_up = _mm256_cmpgt_epi8(filter_16, dst0_up);
            dst0_up = _mm256_and_si256(dst0_up, mask_up);

            dst0 = _mm256_or_si256(dst0, dst0_up);
        }

        {
            __m256i dst0_0 = _mm256_subs_epu8(src0, diff_0);

            __m256i mask_0 = _mm256_cmpgt_epi8(filter_10, dst0_0);
            dst0_0 = _mm256_and_si256(dst0_0, mask_0);

            dst0 = _mm256_or_si256(dst0, dst0_0);
        }


        // 0A 0B 0C 0D  0E 0F 0G 0H ->
        // 00 AB 00 CD  00 EF 00 GH
        __m256i dst1;
        {
            __m256i dst1_lo = _mm256_and_si256(dst0, mask_u16_lo);
            // __m256i mask_u16_hi = _mm256_slli_si256(mask_u16_lo, 1);
            __m256i dst1_hi = _mm256_and_si256(dst0, mask_u16_hi);
                    dst1_lo = _mm256_slli_epi16(dst1_lo, 12);
            
            dst1 = _mm256_or_si256(dst1_lo, dst1_hi);
        }
        // 00 AB 00 CD  00 EF 00 GH ->
        // 00 00 AB CD  00 00 EF GH
        {
            __m256i dst1_lo = _mm256_and_si256(dst1, mask_u32_lo);
            // __m256i mask_u32_hi = _mm256_slli_si256(mask_u32_lo, 2);
            __m256i dst1_hi = _mm256_and_si256(dst1, mask_u32_hi);
            dst1_lo = _mm256_slli_si256(dst1_lo, 1);
            
            dst1 = _mm256_or_si256(dst1_lo, dst1_hi);
        }
        // 00 00 AB CD  00 00 EF GH ->
        // 00 00 00 00  AB CD EF GH
        {
            __m256i dst1_lo = _mm256_and_si256(dst1, mask_u64_lo);
            // __m256i mask_u64_hi = _mm256_slli_si256(mask_u64_lo, 4);
            __m256i dst1_hi = _mm256_and_si256(dst1, mask_u64_hi);
            dst1_lo = _mm256_slli_si256(dst1_lo, 2);
            
            dst1 = _mm256_or_si256(dst1_lo, dst1_hi);
        }
        {
            __m256i dst1_lo = _mm256_and_si256(dst1, mask_u128_lo);
            // __m256i mask_u128_hi = _mm256_slli_si256(mask_u128_lo, 8);
            __m256i dst1_hi = _mm256_and_si256(dst1, mask_u128_hi);
            dst1_lo = _mm256_slli_si256(dst1_lo, 4);
            
            dst1 = _mm256_or_si256(dst1_lo, dst1_hi);
        }

        __m128i dst2_lo = _mm256_extractf128_si256(dst1, 0);
        __m128i dst2_hi = _mm256_extractf128_si256(dst1, 1);
        dst2_lo = _mm_srli_si128(dst2_lo, 8);
        __m128i dst2 = _mm_or_si128(dst2_hi, dst2_lo);

        // *q++ = dst2;
        _mm_store_si128(q++, dst2);
    }

    return base16_any_dec((uint8_t*)q, (uint8_t*)p, input_size % sizeof(__m256i));
}
