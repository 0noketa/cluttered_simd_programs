// 2023-09-08
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <mmintrin.h>

#ifdef __3dNOW__
#define ANY_EMMS _m_femms
#else
#define ANY_EMMS _m_empty
#endif


static const char cs[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/"
    "=";
static const uint_fast8_t cs2[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,62, 0, 0, 0,63,52,53,54,55,56,57,58,59,60,61, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25, 0, 0, 0, 0, 0,
    0,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};


inline __m64 encode_byte_u16x4(__m64 n)
{
    __m64 mask_all = _mm_set1_pi8(0xFF);
    // if (n < 26)
    //     return n + 65;
    __m64 mask0 = _mm_cmpgt_pi16(_mm_set1_pi16(26), n);
    __m64 tmp = _mm_add_pi16(n, _mm_set1_pi16(65));
    __m64 result = _mm_and_si64(tmp, mask0);
    // else if (n < 52)
    //     return n - 26 + 97;
    n = _mm_sub_pi16(n, _mm_set1_pi16(26));
    __m64 mask = _mm_cmpgt_pi16(_mm_set1_pi16(26), n);
    mask = _mm_and_si64(mask, _mm_xor_si64(mask0, mask_all));
    tmp = _mm_add_pi16(n, _mm_set1_pi16(97));
    result = _mm_or_si64(result, _mm_and_si64(tmp, mask));
    mask0 = _mm_or_si64(mask0, mask);
    // else if (n < 62)
    //     return n - 52 + 48;
    n = _mm_sub_pi16(n, _mm_set1_pi16(26));
    mask = _mm_cmpgt_pi16(_mm_set1_pi16(10), n);
    mask = _mm_and_si64(mask, _mm_xor_si64(mask0, mask_all));
    tmp = _mm_add_pi16(n, _mm_set1_pi16(48));
    result = _mm_or_si64(result, _mm_and_si64(tmp, mask));
    mask0 = _mm_or_si64(mask0, mask);
    // else if (n == 62)
    //     return 43;
    n = _mm_sub_pi16(n, _mm_set1_pi16(10));
    mask = _mm_cmpeq_pi16(n, _mm_setzero_si64());
    mask = _mm_and_si64(mask, _mm_xor_si64(mask0, mask_all));
    tmp = _mm_set1_pi16(43);
    result = _mm_or_si64(result, _mm_and_si64(tmp, mask));
    mask0 = _mm_or_si64(mask0, mask);
    // else
    //     return 47;
    mask = _mm_cmpeq_pi16(result, _mm_setzero_si64());
    mask = _mm_and_si64(mask, _mm_xor_si64(mask0, mask_all));
    tmp = _mm_set1_pi16(47);
    result = _mm_or_si64(result, _mm_and_si64(tmp, mask));

    return result;
}


#define PARALLEL

int base64_12n_encode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = input_size / 12;

    const uint32_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    for (size_t i = 0; i < units; ++i)
    {
        uint_fast32_t unit0 = p[i * 3];
        uint_fast32_t unit1 = p[i * 3 + 1];
        uint_fast32_t unit2 = p[i * 3 + 2];
        __m64 tmp;
        __m64 mask0 = _mm_set_pi32(0x00000030, 0x00000003);
        __m64 mask1 = _mm_set_pi32(0x0000000F, 0x0000003C);

        uint_fast32_t  c0 = ( unit0/*&0x000000FC*/>>  2) & 0x3F;
        __m64 cs12;
        cs12 = _mm_set_pi32(unit0 <<  4, unit0 >> 22);
        tmp  = _mm_set_pi32(unit0 >> 12, unit0 >>  6);
        cs12 = _mm_and_si64(cs12, mask0);
        tmp  = _mm_and_si64(tmp, mask1);
        cs12 = _mm_or_si64(cs12, tmp);
        uint_fast32_t  c3 = ((unit0 & 0x003F0000) >> 16);

        uint_fast32_t  c4 = ( unit0/*&0xFC000000*/>> 26);
        __m64 cs56;
        cs56 = _mm_set_pi32(unit0 >> 20, unit1 >> 14);
        tmp  = _mm_set_pi32(unit1 >>  4, unit1 <<  2);
        cs56 = _mm_and_si64(cs56, mask0);
        tmp  = _mm_and_si64(tmp, mask1);
        cs56 = _mm_or_si64(cs56, tmp);
        uint_fast32_t  c7 = ((unit1 & 0x00003F00) >>  8);
#ifdef PARALLEL
        __m64 cs_1526 = _mm_or_si64(cs56, _mm_slli_si64(cs12, 16));
#endif
        uint_fast32_t  c8 = ((unit1 & 0x00FC0000) >> 18);
        __m64 cs9a;
        cs9a = _mm_set_pi32(unit1 >> 12, unit2 >>  6);
        tmp  = _mm_set_pi32(unit1 >> 28, unit1 >> 22);
        cs9a = _mm_and_si64(cs9a, mask0);
        tmp  = _mm_and_si64(tmp, mask1);
        cs9a = _mm_or_si64(cs9a, tmp);
        uint_fast32_t c11 = ((unit2 & 0x0000003F) >>  0);

        uint_fast32_t c12 = ((unit2 & 0x0000FC00) >> 10);
        __m64 csde;
        csde = _mm_set_pi32(unit2 >>  4, unit2 >> 30);
        tmp  = _mm_set_pi32(unit2 >> 20, unit2 >> 14);
        csde = _mm_and_si64(csde, mask0);
        tmp  = _mm_and_si64(tmp, mask1);
        csde = _mm_or_si64(csde, tmp);
        uint_fast32_t c15 = ((unit2 & 0x3F000000) >> 24);
#ifdef PARALLEL
        __m64 cs_9dae = _mm_or_si64(csde, _mm_slli_si64(cs9a, 16));

        cs_1526 = encode_byte_u16x4(cs_1526);
        cs_9dae = encode_byte_u16x4(cs_9dae);

        uint_fast32_t cs_26 = _mm_cvtsi64_si32(cs_1526);
        uint_fast32_t cs_ae = _mm_cvtsi64_si32(cs_9dae);
        cs_1526 = _mm_srli_si64(cs_1526, 32);
        cs_9dae = _mm_srli_si64(cs_9dae, 32);
        uint_fast32_t cs_15 = _mm_cvtsi64_si32(cs_1526);
        uint_fast32_t cs_9d = _mm_cvtsi64_si32(cs_9dae);

        uint_fast32_t  c1 = (cs_15 >> 16) & 0xFF;
        uint_fast32_t  c5 = (cs_15 >>  0) & 0xFF;
        uint_fast32_t  c2 = (cs_26 >> 16) & 0xFF;
        uint_fast32_t  c6 = (cs_26 >>  0) & 0xFF;
        uint_fast32_t  c9 = (cs_9d >> 16) & 0xFF; 
        uint_fast32_t c13 = (cs_9d >>  0) & 0xFF;
        uint_fast32_t c10 = (cs_ae >> 16) & 0xFF;
        uint_fast32_t c14 = (cs_ae >>  0) & 0xFF;
#else
        uint_fast32_t  c2 = _mm_cvtsi64_si32(cs12);
        uint_fast32_t  c6 = _mm_cvtsi64_si32(cs56);
        uint_fast32_t c10 = _mm_cvtsi64_si32(cs9a);
        uint_fast32_t c14 = _mm_cvtsi64_si32(csde);
        cs12 = _mm_srli_si64(cs12, 32);
        cs56 = _mm_srli_si64(cs56, 32);
        cs9a = _mm_srli_si64(cs9a, 32);
        csde = _mm_srli_si64(csde, 32);
        uint_fast32_t  c1 = _mm_cvtsi64_si32(cs12);
        uint_fast32_t  c5 = _mm_cvtsi64_si32(cs56);
        uint_fast32_t  c9 = _mm_cvtsi64_si32(cs9a);
        uint_fast32_t c13 = _mm_cvtsi64_si32(csde);
#endif

        c0 = cs[c0];
#ifndef PARALLEL
        c1 = cs[c1];
        c2 = cs[c2];
#endif
        c3 = cs[c3];

        c4 = cs[c4];
#ifndef PARALLEL
        c5 = cs[c5];
        c6 = cs[c6];
#endif
        c7 = cs[c7];

        c8 = cs[c8];
#ifndef PARALLEL
        c9 = cs[c9];
        c10 = cs[c10];
#endif
        c11 = cs[c11];
        
        c12 = cs[c12];
#ifndef PARALLEL
        c13 = cs[c13];
        c14 = cs[c14];
#endif
        c15 = cs[c15];


        q[i * 4 + 0] = (c0 << 0) | (c1 << 8) | (c2 << 16) | (c3 << 24);
        q[i * 4 + 1] = (c4 << 0) | (c5 << 8) | (c6 << 16) | (c7 << 24);
        q[i * 4 + 2] = (c8 << 0) | (c9 << 8) | (c10 << 16) | (c11 << 24);
        q[i * 4 + 3] = (c12 << 0) | (c13 << 8) | (c14 << 16) | (c15 << 24);
    }


    ANY_EMMS();
    return 1;
}
int base64_24n_encode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base64_12n_encode(input_size - input_size % 24, src, dst);
}
int base64_48n_encode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base64_24n_encode(input_size - input_size % 48, src, dst);
}
int base64_encode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (!base64_12n_encode(input_size, src, dst)) return 0;
    if (src == NULL || dst == NULL) return 0;

    size_t units0 = input_size / 12;
    size_t rem0 = input_size % 12;

    size_t src_base = units0 * 12;
    size_t dst_base = units0 * 16;
    size_t units = rem0 / 3;
    size_t rem = rem0 % 3;

    for (size_t i = 0; i < units; ++i)
    {
        uint32_t buf = src[src_base + i * 3] << 16;
        buf |= src[src_base + i * 3 + 1] << 8;
        buf |= src[src_base + i * 3 + 2];

        dst[dst_base + i * 4 + 0] = cs[(buf >> 18) & 0x3F];
        dst[dst_base + i * 4 + 1] = cs[(buf >> 12) & 0x3F];
        dst[dst_base + i * 4 + 2] = cs[(buf >> 6) & 0x3F];
        dst[dst_base + i * 4 + 3] = cs[(buf >> 0) & 0x3F];
    }

    // 1 11111111 -> 111111 11____ _ _
    // 2 11111111 22222222 -> 111111 112222 2222__ _
    if (rem == 1)
    {
        uint_fast32_t x = src[src_base + units * 3];

        dst[dst_base + units * 4 + 0] = cs[x >> 2];
        dst[dst_base + units * 4 + 1] = cs[(x << 4) & 0x30];
        dst[dst_base + units * 4 + 2] = '=';
        dst[dst_base + units * 4 + 3] = '=';
    }
    else if (rem == 2)
    {
        uint_fast32_t x = src[src_base + units * 3];
        uint_fast32_t y = src[src_base + units * 3 + 1];

        dst[dst_base + units * 4 + 0] = cs[x >> 2];
        dst[dst_base + units * 4 + 0] = cs[((x << 4) | (y >> 4)) & 0x3F];
        dst[dst_base + units * 4 + 0] = cs[(y << 2) & 0x3C];
        dst[dst_base + units * 4 + 0] = '=';
    }

    return 1;
}


int base64_32n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = input_size / 4;

    for (size_t i = 0; i < units; ++i)
    {
        uint_fast32_t c0 = src[i * 4 + 0];
        uint_fast32_t c1 = src[i * 4 + 1];
        uint_fast32_t c2 = src[i * 4 + 2];
        uint_fast32_t c3 = src[i * 4 + 3];
        uint_fast32_t x0 = cs2[c0];
        uint_fast32_t x1 = cs2[c1];
        uint_fast32_t x2 = cs2[c2];
        uint_fast32_t x3 = cs2[c3];

        uint_fast32_t buf = (x0 << 18) | (x1 << 12) | (x2 << 6) | (x3 << 0);

        dst[i * 3 + 0] = (buf >> 16) & 0xFF;
        dst[i * 3 + 1] = (buf >> 8) & 0xFF;
        dst[i * 3 + 2] = (buf >> 0) & 0xFF;
    }

    return 1;
}

int base64_decode(size_t input_size, const uint8_t *src, uint8_t *dst, size_t *out_rem)
{
    size_t units = input_size / 4;
    units -= (units > 0 && input_size % 4 == 0);

    if (!base64_32n_decode(units * 4, src, dst)) return 0;

    size_t result_rem = 0;
    if (units * 4 < input_size)
    {
        uint_fast32_t buf = 0;
        int rem = 0;
        for (size_t i = units * 4; i < input_size; ++i)
        {
            int c = src[i];
            if (c == '=') break;

            int x = cs2[c];
            buf <<= 6;
            buf |= x;
            ++rem;
        }

        size_t j = units * 3;

        buf <<= (4 - rem) * 6;

        if (rem > 0)
        {
            dst[j + 0] = (buf >> 16) & 0xFF;
            ++result_rem;
        }
        if (rem > 1)
        {
            dst[j + 1] = (buf >> 8) & 0xFF;
            ++result_rem;
        }
        if (rem > 2)
        {
            dst[j + 2] = (buf >> 0) & 0xFF;
            ++result_rem;
        }
    }

    if (out_rem) *out_rem = result_rem;

    return 1;
}
