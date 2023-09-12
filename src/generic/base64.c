#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>


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




int base64_24n_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = size / 3;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint32_t buf = src[i * 3] << 16;
        buf |= src[i * 3 + 1] << 8;
        buf |= src[i * 3 + 2];

        dst[i * 4 + 0] = cs[(buf >> 18) & 0x3F];
        dst[i * 4 + 1] = cs[(buf >> 12) & 0x3F];
        dst[i * 4 + 2] = cs[(buf >> 6) & 0x3F];
        dst[i * 4 + 3] = cs[(buf >> 0) & 0x3F];
    }

    return 1;
}
int base64_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    if (!base64_24n_encode(size, src, dst)) return 0;
    if (src == NULL || dst == NULL) return 0;

    size_t units = size / 3;
    size_t rem = size % 3;

    // 1 11111111 -> 111111 11____ _ _
    // 2 11111111 22222222 -> 111111 112222 2222__ _
    if (rem == 1)
    {
        uint_fast32_t x = src[units * 3];

        dst[units * 4 + 0] = cs[x >> 2];
        dst[units * 4 + 1] = cs[(x << 4) & 0x30];
        dst[units * 4 + 2] = '=';
        dst[units * 4 + 3] = '=';
    }
    else if (rem == 2)
    {
        uint_fast32_t x = src[units * 3];
        uint_fast32_t y = src[units * 3 + 1];

        dst[units * 4 + 0] = cs[x >> 2];
        dst[units * 4 + 0] = cs[((x << 4) | (y >> 4)) & 0x3F];
        dst[units * 4 + 0] = cs[(y << 2) & 0x3C];
        dst[units * 4 + 0] = '=';
    }

    return 1;
}


int base64_32n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
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
