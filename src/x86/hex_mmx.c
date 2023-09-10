// 2023-09-08
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>
#include <mmintrin.h>

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif


#define base16_avx_enc base16_avx2_enc
#define base16_avx_dec base16_avx2_dec


static inline __m64 tochars(__m64 src, __m64 diff_a);


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


// void base16_avx_enc(
//     uint8_t *dst,
//     const uint8_t *src,
//     size_t input_size, bool upper)
// {
//     if (src == NULL || dst == NULL)
//         return;

//     __m64 diff_a_upper = _mm_set_pi32(0x31313131, 0x31313131);
//     __m64 diff_a_lower = _mm_set_pi32(0x57575757, 0x57575757);
//     __m64 diff_a = upper ? diff_a_upper : diff_a_lower;

//     size_t units = input_size / sizeof(__m64);
//     size_t rest = input_size % sizeof(__m64);

//     const __m64 *p = (void*) src;
//     __m64 *q = (void*) dst;

//     for (size_t i = 0; i < units; ++i)
//     {
//         // 67452301 (ex in v4i8) -> 06040200, 07050301
//         __m64 src0 = *p++;
//         __m64 dst0_hi = _mm_srli_pi64(src0, 4); 
//         __m64 mask_0f = _mm_set_pi32(0x0F0F0F0F, 0x0F0F0F0F);
//         dst0_hi = _mm_and_si64(dst0_hi, mask_0f);
//         __m64 dst0_lo = _mm_and_si64(src0, mask_0f);

//         // 07050301, 06040200 -> 07060302, 05040100
//         __m64 dst1_lo = _mm256_unpacklo_pi8(dst0_hi, dst0_lo);
//         __m64 dst1_hi = _mm256_unpackhi_pi8(dst0_hi, dst0_lo);

//         __m128i x = _mm256_extracti128_si256(dst1_lo, 1);
//         __m128i y = _mm256_extracti128_si256(dst1_hi, 0);
//         __m64 dst2_lo = _mm256_inserti128_si256(dst1_lo, y, 1);
//         __m64 dst2_hi = _mm256_inserti128_si256(dst1_hi, x, 0);

//         __m64 dst_hi = tochars(dst2_hi, diff_a);
//         __m64 dst_lo = tochars(dst2_lo, diff_a);

//         _mm256_store_si256(q++, dst_lo);
//         _mm256_store_si256(q++, dst_hi);
//     }

//     const uint8_t *src_r = (void*) p;
//     uint8_t *dst_r = (void*) q;

//     base16_any_enc(dst_r, src_r, rest, upper);
// }

static inline __m64 tochars(__m64 src, __m64 diff_a)
{ 
    __m64 diff_0 = _mm_set_pi32(0x30303030, 0x30303030);
    __m64 filter_10 = _mm_set_pi32(0x0A0A0A0A, 0x0A0A0A0A);
    __m64 filter_9 = _mm_set_pi32(0x09090909, 0x09090909);

    // [int4] -> _0: [int4 + '0'], _a: [int4 + 'A' - 10]
    __m64 dst0_0 = _mm_adds_pu8(src, diff_0);
    __m64 dst0_a = _mm_adds_pu8(src, diff_a);
    __m64 mask_0 = _mm_cmpgt_pi8(filter_10, src);
    __m64 mask_a = _mm_cmpgt_pi8(src, filter_9);
    __m64 dst_0 = _mm_and_si64(dst0_0, mask_0);
    __m64 dst_a = _mm_and_si64(dst0_a, mask_a);

    __m64 dst = _mm_or_si64(dst_0, dst_a);

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


int base16_128n_decode(size_t input_size, const uint32_t *src, uint32_t *dst)
{
    size_t units = input_size / sizeof(__m64);

    const __m64 *p = (void*)src;
    uint32_t *q = dst;

    for (size_t i = 0; i < units; ++i)
    {
        __m64 src0 = *p++;

        __m64 dst0;
        {
            __m64 diff_a_lower = _mm_set_pi32(0x57575757, 0x57575757);
            __m64 dst0_lo = _mm_subs_pu8(src0, diff_a_lower);

            __m64 filter_lower = _mm_set_pi32(0x60606060, 0x60606060);
            __m64 mask_lo = _mm_cmpgt_pi8(src0, filter_lower);
            dst0_lo = _mm_and_si64(dst0_lo, mask_lo);

            __m64 filter_16 = _mm_set_pi32(0x10101010, 0x10101010);
            mask_lo = _mm_cmpgt_pi8(filter_16, dst0_lo);
            dst0_lo = _mm_and_si64(dst0_lo, mask_lo);

            dst0 = dst0_lo;
        }

        {
            __m64 diff_a_upper = _mm_set_pi32(0x37373737, 0x37373737);
            __m64 dst0_up = _mm_subs_pi8(src0, diff_a_upper);

            __m64 filter_16 = _mm_set_pi32(0x10101010, 0x10101010);
            __m64 mask_up = _mm_cmpgt_pi8(filter_16, dst0_up);
            dst0_up = _mm_and_si64(dst0_up, mask_up);

            __m64 filter_upper = _mm_slli_si64(filter_16, 2);
            // __m64 filter_upper = _mm_set_pi32(0x40404040, 0x40404040);
            mask_up = _mm_cmpgt_pi8(src0, filter_upper);
            dst0_up = _mm_and_si64(dst0_up, mask_up);

            dst0 = _mm_or_si64(dst0, dst0_up);
        }

        {
            __m64 diff_0 = _mm_set_pi32(0x30303030, 0x30303030);
            __m64 dst0_0 = _mm_subs_pu8(src0, diff_0);

            __m64 filter_10 = _mm_set_pi32(0x0A0A0A0A, 0x0A0A0A0A);
            __m64 mask_0 = _mm_cmpgt_pi8(filter_10, dst0_0);
            dst0_0 = _mm_and_si64(dst0_0, mask_0);

            dst0 = _mm_or_si64(dst0, dst0_0);
        }


        // 0A 0B 0C 0D  0E 0F 0G 0H ->
        // 00 AB 00 CD  00 EF 00 GH
        __m64 dst1;
        {
            __m64 mask_u16_hi = _mm_set_pi32(0xFF00FF00, 0xFF00FF00);
            __m64 dst1_hi = _mm_and_si64(dst0, mask_u16_hi);
            // __m64 mask_u16_lo = _mm_set_pi32(0x00FF00FF, 0x00FF00FF);
            __m64 mask_u16_lo = _mm_srli_si64(mask_u16_hi, 8);
            __m64 dst1_lo = _mm_and_si64(dst0, mask_u16_lo);
            dst1_lo = _mm_slli_pi16(dst1_lo, 12);
            
            dst1 = _mm_or_si64(dst1_lo, dst1_hi);
        }
        // 00 AB 00 CD  00 EF 00 GH ->
        // 00 00 AB CD  00 00 EF GH
        {
            __m64 mask_u32_lo = _mm_set_pi32(0x0000FFFF, 0x0000FFFF);
            __m64 dst1_lo = _mm_and_si64(dst1, mask_u32_lo);
            // __m64 mask_u32_hi = _mm_set_pi32(0xFFFF0000, 0xFFFF0000);
            __m64 mask_u32_hi = _mm_slli_si64(mask_u32_lo, 16);
            __m64 dst1_hi = _mm_and_si64(dst1, mask_u32_hi);
            dst1_lo = _mm_slli_si64(dst1_lo, 8);
            
            dst1 = _mm_or_si64(dst1_lo, dst1_hi);
        }
        // 00 00 AB CD  00 00 EF GH ->
        // 00 00 00 00  AB CD EF GH
        {
            __m64 mask_u64_lo = _mm_set_pi32(0x00000000, 0xFFFFFFFF);
            __m64 dst1_lo = _mm_and_si64(dst1, mask_u64_lo);
            // __m64 mask_u64_hi = _mm_set_pi32(0xFFFFFFFF, 0x00000000);
            __m64 mask_u64_hi = _mm_slli_si64(mask_u64_lo, 32);
            __m64 dst1_hi = _mm_and_si64(dst1, mask_u64_hi);
            dst1_lo = _mm_slli_si64(dst1_lo, 16);

            dst1 = _mm_or_si64(dst1_lo, dst1_hi);
        }

        dst1 = _mm_srli_si64(dst1, 32);
        uint32_t dst2 = _mm_cvtsi64_si32(dst1);

        *q++ = dst2;
    }


    ANY_EMMS();

    return base16_any_dec((uint8_t*)q, (uint8_t*)p, input_size % sizeof(__m64));
}
