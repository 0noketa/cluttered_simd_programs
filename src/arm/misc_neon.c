#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

#include "../include/simd_tools.h"

#ifdef _MSC_VER
#define INIT_I16X4(n) { .n64_i16 = { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) } }
#define INIT_I8X8(n) { .n64_i8 = { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) } }
#else
#define INIT_I16X4(n) { (int16_t)(n), (int16_t)(n), (int16_t)(n), (int16_t)(n) }
#define INIT_I8X8(n) { (int8_t)(n), (int8_t)(n), (int8_t)(n), (int8_t)(n) }
#endif

/* local */

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


/* humming weight */

size_t vec_u256n_get_humming_weight(size_t size, uint8_t *src)
{
    size_t units = size / 16;

    __m128i *p = (__m128i*)src;

    __m128i rs = _mm_setzero_si128();

    for (size_t i = 0; i < units; ++i)
    {
        __m128i it = p[i++];
        __m128i it2 = p[i];
        __m128i rs0 = _mm_setzero_si128();

        __m128i one_x16 = _mm_set1_epi8(1);

        for (int j = 0; j < 8; ++j)
        {
            __m128i tmp = _mm_and_si128(it, one_x16);
            __m128i tmp2 = _mm_and_si128(it2, one_x16);

            rs0 = _mm_adds_epi8(rs0, tmp);
            rs0 = _mm_adds_epi8(rs0, tmp2);

            it = _mm_srli_epi32(it, 1);
            it2 = _mm_srli_epi32(it2, 1);
         }

        __m128i mask = _mm_set1_epi32(0x000000FF);
        __m128i mask2 = _mm_set1_epi32(0x0000FF00);
        __m128i mask3 = _mm_set1_epi32(0x00FF0000);
        __m128i mask4 = _mm_set1_epi32(0xFF000000);

        __m128i masked = _mm_and_si128(mask, rs0);
        __m128i masked2 = _mm_and_si128(mask2, rs0);
        __m128i masked3 = _mm_and_si128(mask3, rs0);
        __m128i masked4 = _mm_and_si128(mask4, rs0);
        masked2 = _mm_srli_si128(masked2, 1);
        masked3 = _mm_srli_si128(masked3, 2);
        masked4 = _mm_srli_si128(masked4, 3);

        masked = _mm_add_epi32(masked, masked3);
        masked2 = _mm_add_epi32(masked2, masked4);

        rs = _mm_add_epi32(rs, masked);
        rs = _mm_add_epi32(rs, masked2);
    }

    size_t r0 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r1 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r2 = _mm_cvtsi128_si32(rs);
    rs = _mm_srli_si128(rs, 4);
    size_t r3 = _mm_cvtsi128_si32(rs);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}
