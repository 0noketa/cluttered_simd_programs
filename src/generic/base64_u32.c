#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>


static const char encode_char[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/"
    "=";

#ifdef USE_LUT2
#include "./base64_12bit_encoder.c"
#endif

#if defined USE_LUT || defined USE_LUT2
static const uint_fast8_t decode_char_[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,62, 0, 0, 0,63,52,53,54,55,56,57,58,59,60,61, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25, 0, 0, 0, 0, 0,
    0,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51, 0, 0, 0, 0, 0,

    0,
};
#define decode_char(c) decode_char_[c]
#else
static uint32_t decode_char(uint32_t c)
{
    return isupper(c) ? c - 'A'
            : islower(c) ? c - 'a' - 26
            : isdigit(c) ? c - '0' + 52
            : c == '+' ? 53
            : c == '/' ? 54
            : 0;
}
#endif



int base64_48n_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    return base64_24n_encode(size - size % 48, src, dst);
}
int base64_24n_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    return base64_12n_encode(size - size % 24, src, dst);
}
int base64_12n_encode(size_t size, const uint8_t *src, uint8_t *dst)
#ifdef USE_LUT2
{
    return base64_3n_encode(size - size % 12, src, dst);
}
#else
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = size / 12;

    const uint32_t *p = (void*)src;
    uint32_t *q = (void*)dst;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
#ifdef __AVX2__
#   ifdef _MSC_VER
        uint_fast32_t unit0 = _load_be_u32(p + i * 3);
        uint_fast32_t unit1 = _load_be_u32(p + i * 3 + 1);
        uint_fast32_t unit2 = _load_be_u32(p + i * 3 + 2);
#   else
        uint_fast32_t unit0 = __builtin_bswap32(p[i * 3]);
        uint_fast32_t unit1 = __builtin_bswap32(p[i * 3 + 1]);
        uint_fast32_t unit2 = __builtin_bswap32(p[i * 3 + 2]);
#   endif
#else
        uint_fast32_t unit0 = p[i * 3];
        uint_fast32_t unit1 = p[i * 3 + 1];
        uint_fast32_t unit2 = p[i * 3 + 2];
#endif

#if _BYTE_ORDER == _LITTLE_ENDIAN && !defined(__AVX2__)
        uint_fast32_t  c0 = ( unit0/*&0x000000FC*/>>  2) & 0x3F;
        uint_fast32_t  c1 = ((unit0 & 0x00000003) <<  4) | ((unit0 & 0x0000F000) >> 12);
        uint_fast32_t  c2 = ((unit0 & 0x00C00000) >> 22) | ((unit0 & 0x00000F00) >>  6);
        uint_fast32_t  c3 = ((unit0 & 0x003F0000) >> 16);

        uint_fast32_t  c4 = ( unit0/*&0xFC000000*/>> 26);
        uint_fast32_t  c5 = ((unit0 & 0x03000000) >> 20) | ((unit1 & 0x000000F0) >>  4);
        uint_fast32_t  c6 = ((unit1 & 0x0000C000) >> 14) | ((unit1 & 0x0000000F) <<  2);
        uint_fast32_t  c7 = ((unit1 & 0x00003F00) >>  8);

        uint_fast32_t  c8 = ((unit1 & 0x00FC0000) >> 18);
        uint_fast32_t  c9 = ((unit1 & 0x00030000) >> 12) | ( unit1/*&0xF0000000*/>> 28);
        uint_fast32_t c10 = ((unit2 & 0x000000C0) >>  6) | ((unit1 & 0x0F000000) >> 22);
        uint_fast32_t c11 = ((unit2 & 0x0000003F) >>  0);

        uint_fast32_t c12 = ((unit2 & 0x0000FC00) >> 10);
        uint_fast32_t c13 = ((unit2 & 0x00000300) >>  4) | ((unit2 & 0x00F00000) >> 20);
        uint_fast32_t c14 = ( unit2/*&0xC0000000*/>> 30) | ((unit2 & 0x000F0000) >> 14);
        uint_fast32_t c15 = ((unit2 & 0x3F000000) >> 24);
#else
        uint_fast32_t  c0 = (unit0 >> 26);
        uint_fast32_t  c1 = (unit0 >> 20) & 0x3F;
        uint_fast32_t  c2 = (unit0 >> 14) & 0x3F;
        uint_fast32_t  c3 = (unit0 >>  8) & 0x3F;

        uint_fast32_t  c4 = (unit0 >>  2) & 0x3F;
        uint_fast32_t  c5 = ((unit0 <<  4) | (unit1 >> 28)) & 0x3F;
        uint_fast32_t  c6 = (unit1 >> 22) & 0x3F;
        uint_fast32_t  c7 = (unit1 >> 16) & 0x3F;

        uint_fast32_t  c8 = (unit1 >> 10) & 0x3F;
        uint_fast32_t  c9 = (unit1 >> 4) & 0x3F;
        uint_fast32_t c10 = ((unit1 <<  2) | (unit2 >> 30)) & 0x3F;
        uint_fast32_t c11 = (unit2 >> 24) & 0x3F;

        uint_fast32_t c12 = (unit2 >> 18) & 0x3F;
        uint_fast32_t c13 = (unit2 >> 12) & 0x3F;
        uint_fast32_t c14 = (unit2 >>  6) & 0x3F;
        uint_fast32_t c15 = (unit2 >>  0) & 0x3F;
#endif

        c0 = encode_char[c0];
        c1 = encode_char[c1];
        c2 = encode_char[c2];
        c3 = encode_char[c3];

        c4 = encode_char[c4];
        c5 = encode_char[c5];
        c6 = encode_char[c6];
        c7 = encode_char[c7];

        c8 = encode_char[c8];
        c9 = encode_char[c9];
        c10 = encode_char[c10];
        c11 = encode_char[c11];
        
        c12 = encode_char[c12];
        c13 = encode_char[c13];
        c14 = encode_char[c14];
        c15 = encode_char[c15];

#if _BYTE_ORDER == _LITTLE_ENDIAN
        q[i * 4 + 0] = (c0 << 0) | (c1 << 8) | (c2 << 16) | (c3 << 24);
        q[i * 4 + 1] = (c4 << 0) | (c5 << 8) | (c6 << 16) | (c7 << 24);
        q[i * 4 + 2] = (c8 << 0) | (c9 << 8) | (c10 << 16) | (c11 << 24);
        q[i * 4 + 3] = (c12 << 0) | (c13 << 8) | (c14 << 16) | (c15 << 24);
#else
        q[i * 4 + 0] = (c0 << 24) | (c1 << 16) | (c2 << 8) | (c3 << 0);
        q[i * 4 + 1] = (c4 << 24) | (c5 << 16) | (c6 << 8) | (c7 << 0);
        q[i * 4 + 2] = (c8 << 24) | (c9 << 16) | (c10 << 8) | (c11 << 0);
        q[i * 4 + 3] = (c12 << 24) | (c13 << 16) | (c14 << 8) | (c15 << 0);
#endif
    }

    return 1;
}
#endif
int base64_3n_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 3;

    uint32_t *q = (void*)dst;

    int i = 0;
#ifdef USE_LUT2
    #pragma omp parallel for num_threads(4)
#endif
    for (i = 0; i < units; ++i)
    {
        uint32_t unit = src[i * 3] << 16;
        unit |= src[i * 3 + 1] << 8;
        unit |= src[i * 3 + 2];

#ifdef USE_LUT2
#   if _BYTE_ORDER == _LITTLE_ENDIAN
        uint32_t unit2 = base64_12bit_encode_[unit & 0x0FFF] << 16;
        unit2 |= base64_12bit_encode_[(unit >> 12) & 0x0FFF];
#   else
        uint32_t unit2 = base64_12bit_encode_[unit & 0x0FFF];
        unit2 |= base64_12bit_encode_[(unit >> 12) & 0x0FFF] << 16;
#   endif
        q[i] = unit2;
#else
        dst[i * 4 + 0] = encode_char[(unit >> 18) & 0x3F];
        dst[i * 4 + 1] = encode_char[(unit >> 12) & 0x3F];
        dst[i * 4 + 2] = encode_char[(unit >> 6) & 0x3F];
        dst[i * 4 + 3] = encode_char[(unit >> 0) & 0x3F];
#endif
    }

    return 1;
}
int base64_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    if (!base64_12n_encode(size, src, dst)) return 0;
    if (src == NULL || dst == NULL) return 0;

    size_t units0 = size / 12;
    size_t src_base0 = units0 * 12;
    size_t dst_base0 = units0 * 16;

    if (!base64_3n_encode(size % 12, src + src_base0, dst + dst_base0)) return 0;

    size_t units = size / 3;
    size_t src_base = units * 3;
    size_t dst_base = units * 4;
    size_t rem = size % 3;

    // 1 11111111 -> 111111 11____ _ _
    // 2 11111111 22222222 -> 111111 112222 2222__ _
    if (rem == 1)
    {
        uint_fast32_t x = src[src_base + units * 3];

        dst[dst_base + 0] = encode_char[x >> 2];
        dst[dst_base + 1] = encode_char[(x << 4) & 0x30];
        dst[dst_base + 2] = '=';
        dst[dst_base + 3] = '=';
    }
    else if (rem == 2)
    {
        uint_fast32_t x = src[src_base];
        uint_fast32_t y = src[src_base + 1];

        dst[dst_base + 0] = encode_char[x >> 2];
        dst[dst_base + 1] = encode_char[((x << 4) | (y >> 4)) & 0x3F];
        dst[dst_base + 2] = encode_char[(y << 2) & 0x3C];
        dst[dst_base + 3] = '=';
    }

    return 1;
}


int base64_32n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    return base64_4n_decode(input_size - input_size % 32, src, dst);
}

int base64_4n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = input_size / 4;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint_fast32_t c0 = src[i * 4 + 0];
        uint_fast32_t c1 = src[i * 4 + 1];
        uint_fast32_t c2 = src[i * 4 + 2];
        uint_fast32_t c3 = src[i * 4 + 3];
        uint_fast32_t x0 = decode_char(c0);
        uint_fast32_t x1 = decode_char(c1);
        uint_fast32_t x2 = decode_char(c2);
        uint_fast32_t x3 = decode_char(c3);

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

    if (!base64_4n_decode(units * 4, src, dst)) return 0;

    size_t result_rem = 0;
    if (units * 4 < input_size)
    {
        uint_fast32_t buf = 0;
        int rem = 0;
        for (size_t i = units * 4; i < input_size; ++i)
        {
            int c = src[i];
            if (c == '=') break;

            int x = decode_char(c);
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
