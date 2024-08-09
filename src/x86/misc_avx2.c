// #define _M_IX86
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#if defined(__GNUC__)
#include <x86intrin.h>
#elif defined(_MSC_VER) 
#include <intrin.h>
#endif

#include "../../include/simd_tools.h"


/* local */
#if 0
#include <stdio.h>
static void dump(const char *s, __m256i current)
{
    int it;
   fputs(s, stdout);
 
    #define template_block(n) \
    { \
        it = _mm256_extract_epi32(current, n); \
        \
        for (int i = 0; i < 2; ++i) \
        { \
            printf("%d,", (int)(int16_t)(it & UINT16_MAX)); \
            it >>= 16; \
        } \
    }
    template_block(0)
    template_block(1)
    template_block(2)
    template_block(3)
    template_block(4)
    template_block(5)
    template_block(6)
    template_block(7)

    fputs("\n", stdout);
}
static void dump32_128(const char *s, __m128i current)
{
    fputs(s, stdout);
 
    for (int i = 0; i < 4; ++i)
    {
        int32_t it = _mm_cvtsi128_si32(current);
        current = _mm_srli_si128(current, 4);
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump16_128(const char *s, __m128i current)
{
    fputs(s, stdout);
 
    for (int i = 0; i < 8; ++i)
    {
        int16_t it = _mm_cvtsi128_si32(current) & 0xFFFF;
        current = _mm_srli_si128(current, 2);
        printf("%d,", (int)it);
    }

    fputs("\n", stdout);
}
static void dump8(const char *s, __m256i current)
{

    fputs(s, stdout);

    int8_t *p = (int8_t*)&current;
    for (int i = 0; i < 32; ++i)
    {
        printf("%d,", p[i]);
    }

    fputs("\n", stdout);
}
static void dump8_128(const char *s, __m128i current)
{

    fputs(s, stdout);

    int8_t *p = (int8_t*)&current;
    for (int i = 0; i < 16; ++i)
    {
        printf("%d,", p[i]);
    }

    fputs("\n", stdout);
}
#endif





void  vec_i16x16n_abs(size_t size, int16_t *src)
{
    size_t units = size / 16;
	
	__m256i *p = (__m256i*)src;
	__m256i umax_x16 = _mm256_set1_epi16(UINT16_MAX);
	__m256i zero_x16 = _mm256_setzero_si256();

	for (size_t i = 0; i < units; ++i)
	{
		__m256i it = p[i];
		
		__m256i positive_mask = _mm256_cmpgt_epi16(it, umax_x16);
		__m256i neg_mask = _mm256_xor_si256(positive_mask, umax_x16);
		__m256i modified = _mm256_sub_epi16(zero_x16, it);
		it = _mm256_and_si256(it, positive_mask);
		modified = _mm256_and_si256(modified, neg_mask);
		p[i] = _mm256_or_si256(it, modified);
	}
}

void  vec_i16x16n_diff(size_t size, int16_t *base, int16_t *target, int16_t *dst)
;



void  vec_i16x16n_dist(size_t size, const int16_t *src1, const int16_t *src2, int16_t *dst)
{
	size_t units = size / 16;
	__m256i *p = (__m256i*)src1;
	__m256i *q = (__m256i*)src2;
	__m256i *r = (__m256i*)dst;

	__m256i umax_x16 = _mm256_set1_epi16(UINT16_MAX);
	__m256i min_x16 = _mm256_set1_epi16(INT16_MIN);
	__m256i max_x16 = _mm256_set1_epi16(INT16_MAX);


	for (size_t i = 0; i < units; ++i)
	{
		__m256i it = _mm256_subs_epi16(q[i], p[i]);

		__m256i mask_min = _mm256_cmpeq_epi16(it, min_x16);
		__m256i mask_positive = _mm256_cmpgt_epi16(it, umax_x16);

		__m256i zero_x16 = _mm256_setzero_si256();
		__m256i notmin_it = _mm256_sub_epi16(zero_x16, it);
		notmin_it = _mm256_andnot_si256(mask_positive, notmin_it);
		it = _mm256_and_si256(it, mask_positive);
		notmin_it = _mm256_or_si256(notmin_it, it);
		notmin_it = _mm256_andnot_si256(mask_min, notmin_it);

		/*
		 * result = (mask_min & max_x16)   # -> it''
		 * 		| (mask_notmin
		 * 			& ( (mask_negative  & (0 - it))  # notmin_it -> notmin_it' -> it''
		 * 				| (mask_positive & it)  # it' -> notmin_it' -> it''
		 * 				)
		 * 		 )
		 */
		it = _mm256_and_si256(mask_min, max_x16);  // min->max
		it = _mm256_or_si256(it, notmin_it);

		r[i] = it;
	}
}


/* hamming weight */

static inline size_t vec_u256n_get_hamming_weight_popcnt(size_t size, const uint8_t *src)
{
#if defined(_WIN64) || defined(__X86_64__)
    size_t units = size / 8;
    uint64_t *p = (uint64_t*)src;
#else
    size_t units = size / 4;
    uint32_t *p = (uint32_t*)src;
#endif

    size_t r = 0;

    for (size_t i = 0; i < units; ++i)
    {
#if defined(_WIN64)
        r += __popcnt64(p[i]);
#elif defined(__X86_64__)
        r += __popcntq(p[i]);
#elif defined(_WIN32) && defined(_MSC_VER)
        r += __popcnt(p[i]);
#else
        r += __popcntd(p[i]);
#endif
    }

    return r;
}
static inline __m256i vec_u256n_get_hamming_weight_i(__m256i it, __m256i it2, __m256i rs)
{
    __m256i mask = _mm256_set1_epi32(0x55555555);
    __m256i mask2 = _mm256_set1_epi32(0x33333333);
    __m256i mask3 = _mm256_set1_epi32(0x0F0F0F0F);

    __m256i tmp = _mm256_srli_epi32(it, 1);
    __m256i tmp2 = _mm256_srli_epi32(it2, 1);
    it = _mm256_and_si256(it, mask);
    it2 = _mm256_and_si256(it2, mask);
    tmp = _mm256_and_si256(tmp, mask);
    tmp2 = _mm256_and_si256(tmp2, mask);
    it = _mm256_adds_epu16(it, tmp);
    it2 = _mm256_adds_epu16(it2, tmp2);

    tmp = _mm256_srli_epi32(it, 2);
    tmp2 = _mm256_srli_epi32(it2, 2);
    it = _mm256_and_si256(it, mask2);
    it2 = _mm256_and_si256(it2, mask2);
    tmp = _mm256_and_si256(tmp, mask2);
    tmp2 = _mm256_and_si256(tmp2, mask2);
    it = _mm256_adds_epu16(it, tmp);
    it2 = _mm256_adds_epu16(it2, tmp2);

    tmp = _mm256_srli_epi32(it, 4);
    tmp2 = _mm256_srli_epi32(it2, 4);
    it = _mm256_and_si256(it, mask3);
    it2 = _mm256_and_si256(it2, mask3);
    tmp = _mm256_and_si256(tmp, mask3);
    tmp2 = _mm256_and_si256(tmp2, mask3);
    it = _mm256_add_epi32(it, tmp);
    it2 = _mm256_add_epi32(it2, tmp2);

    it = _mm256_add_epi32(it, it2);

    __m256i bytemask = _mm256_set1_epi32(0x000000FF);
    __m256i bytemask2 = _mm256_set1_epi32(0x0000FF00);
    __m256i bytemask3 = _mm256_set1_epi32(0x00FF0000);
    __m256i bytemask4 = _mm256_set1_epi32(0xFF000000);

    __m256i masked = _mm256_and_si256(bytemask, it);
    __m256i masked2 = _mm256_and_si256(bytemask2, it);
    __m256i masked3 = _mm256_and_si256(bytemask3, it);
    __m256i masked4 = _mm256_and_si256(bytemask4, it);
    masked2 = _mm256_srli_si256(masked2, 1);
    masked3 = _mm256_srli_si256(masked3, 2);
    masked4 = _mm256_srli_si256(masked4, 3);

    masked = _mm256_add_epi32(masked, masked3);
    masked2 = _mm256_add_epi32(masked2, masked4);

    rs = _mm256_add_epi32(rs, masked);
    rs = _mm256_add_epi32(rs, masked2);

    return rs;
}
size_t vec_u256n_get_hamming_weight(size_t size, const uint8_t *src)
#if defined(_WIN64) || defined(__X86_64__) || defined(USE_POPCNT)
{
    return vec_u256n_get_hamming_weight_popcnt(size, src);
}
#else
{
    if (size % 64 != 0)
    {
        return vec_u256n_get_hamming_weight_popcnt(size, src);
    }

    size_t units = size / 32 / 2;

    __m256i *p = (__m256i*)src;

    __m256i rs = _mm256_setzero_si256();

    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i * 2];
        __m256i it2 = p[i * 2 + 1];

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs);
    }

    __m128i rs0 = _mm256_extracti128_si256(rs, 0);
    __m128i rs1 = _mm256_extracti128_si256(rs, 1);
    rs0 = _mm_add_epi32(rs0, rs1);

    size_t r0 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r1 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r2 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r3 = _mm_cvtsi128_si32(rs0);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}
#endif

static inline size_t vec_u256n_get_hamming_distance_popcnt(size_t size, const uint8_t *src1, const uint8_t *src2)
{
#if defined(_WIN64) || defined(__X86_64__)
    size_t units = size / 8;
    uint64_t *p = (uint64_t*)src1;
    uint64_t *q = (uint64_t*)src2;
#else
    size_t units = size / 4;
    uint32_t *p = (uint32_t*)src1;
    uint32_t *q = (uint32_t*)src2;
#endif

    size_t r = 0;

    for (size_t i = 0; i < units; ++i)
    {
#if defined(_WIN64)
        r += __popcnt64(p[i] ^ q[i]);
#elif defined(__X86_64__)
        r += __popcntq(p[i] ^ q[i]);
#elif defined(_WIN32) && defined(_MSC_VER)
        r += __popcnt(p[i] ^ q[i]);
#else
        r += __popcntd(p[i] ^ q[i]);
#endif
    }

    return r;
}
size_t vec_u256n_get_hamming_distance(size_t size, const uint8_t *src1, const uint8_t *src2)
#if defined(_WIN64) || defined(__X86_64__) || defined(USE_POPCNT)
{
    return vec_u256n_get_hamming_distance_popcnt(size, src1, src2);
}
#else
{
    if (size % 64 != 0)
    {
        return vec_u256n_get_hamming_distance_popcnt(size, src1, src2);
    }

    size_t units = size / 32 / 2;

    __m256i *p = (__m256i*)src1;
    __m256i *q = (__m256i*)src2;

    __m256i rs = _mm256_setzero_si256();

    for (size_t i = 0; i < units; ++i)
    {
        __m256i it = p[i * 2];
        __m256i it2 = p[i * 2 + 1];
        __m256i it_b = q[i * 2];
        __m256i it2_b = q[i * 2 + 1];

        it = _mm256_xor_si256(it, it_b);
        it2 = _mm256_xor_si256(it2, it2_b);

        rs = vec_u256n_get_hamming_weight_i(it, it2, rs);
    }

    __m128i rs0 = _mm256_extracti128_si256(rs, 0);
    __m128i rs1 = _mm256_extracti128_si256(rs, 1);
    rs0 = _mm_add_epi32(rs0, rs1);

    size_t r0 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r1 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r2 = _mm_cvtsi128_si32(rs0);
    rs0 = _mm_srli_si128(rs0, 4);
    size_t r3 = _mm_cvtsi128_si32(rs0);

    size_t r = r0 + r1 + r2 + r3;

    return r;
}
#endif


int32_t vec_i32x8n_avg(size_t size, const int32_t *src)
{
    return vec_i32x8n_sum_i32(size, src) / size;
}
double vec_i32x8n_avg_f64(size_t size, const int32_t *src)
{
    return ((double)vec_i32x8n_sum_i32(size, src) / size);
}
int16_t vec_i16x16n_avg(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum_i16(size, src) / size;
}
float vec_i16x16n_avg_f32(size_t size, const int16_t *src)
{
    return (float)((double)vec_i16x16n_sum_i16(size, src) / size);
}
int8_t vec_i8x32n_avg(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum_i32(size, src) / size;
}
float vec_i8x32n_avg_f32(size_t size, const int8_t *src)
{
    return (float)((double)vec_i8x32n_sum_i32(size, src) / size);
}


/* sum */

int32_t vec_i32x8n_sum_i32(size_t size, const int32_t *src)
{
    size_t units = size / 8;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    int32_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    results = _mm_srli_si128(results, 4);
    int32_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
uint32_t vec_u32v8n_sum_u32(size_t size, const uint32_t *src)
;

int16_t vec_i16x16n_sum_i16(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_adds_epi16(results, it);
        results2 = _mm_adds_epi16(results2, it2);
    }

    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi16(results, results2);

    int16_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
int32_t vec_i16x16n_sum_i32(size_t size, const int16_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];
        __m128i flags = _mm_srai_epi16(it, 15);
        __m128i flags2 = _mm_srai_epi16(it2, 15);
        __m128i it3 = _mm_unpackhi_epi16(it, flags);
        __m128i it4 = _mm_unpackhi_epi16(it2, flags2);
        it = _mm_unpacklo_epi16(it, flags);
        it2 = _mm_unpacklo_epi16(it2, flags2);

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    return result;
}
size_t vec_i16x16n_sum(size_t size, const int16_t *src)
{
    return vec_i16x16n_sum_i32(size, src);
}

// returns u32x4
static __m128i vec_u16v16_sum_i32x4(size_t size, const uint8_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        __m128i it_hi = _mm_slli_epi32(it, 2);
        __m128i it2_hi = _mm_slli_epi32(it2, 2);
        __m128i it_lo = _mm_and_si128(it, mask_lower);
        __m128i it2_lo = _mm_and_si128(it2, mask_lower);

        results = _mm_add_epi32(results, it_hi);
        results2 = _mm_add_epi32(results2, it2_hi);
        results = _mm_add_epi32(results, it_lo);
        results2 = _mm_add_epi32(results2, it2_lo);
    }

    return _mm_add_epi32(results, results2);
}
uint16_t vec_u16v16n_sum_u16(size_t size, const uint16_t *src)
;
size_t vec_u16v16n_sum(size_t size, const uint16_t *src)
;

int8_t vec_i8x32n_sum_i8(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        results = _mm_adds_epi8(results, it);
        results2 = _mm_adds_epi8(results2, it2);
    }

    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi8(results, results2);

    results2 = _mm_srli_si128(results, 1);
    results = _mm_adds_epi8(results, results2);

    int8_t result = _mm_cvtsi128_si32(results) & 0xFF;
    return result;
}
int16_t vec_i8x32n_sum_i16(size_t size, const int8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];
        __m128i it3 = _mm_cvtepi8_epi16(it);
        __m128i it4 = _mm_cvtepi8_epi16(it2);
        it2 = _mm_srli_si128(it2, 8);
        it = _mm_srli_si128(it, 8);
        it = _mm_cvtepi8_epi16(it);
        it2 = _mm_cvtepi8_epi16(it2);

        results = _mm_adds_epi16(results, it);
        results2 = _mm_adds_epi16(results2, it2);
    }

    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_adds_epi16(results, results2);

    results2 = _mm_srli_si128(results, 2);
    results = _mm_adds_epi16(results, results2);

    int16_t result = _mm_cvtsi128_si32(results) & 0xFFFF;
    return result;
}
int32_t vec_i8x32n_sum_i32(size_t size, const int8_t *src)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        __m128i it2 = _mm_cvtepi8_epi16(it);
        it = _mm_srli_si128(it, 8);
        it = _mm_cvtepi8_epi16(it);
 
        __m128i it3 = _mm_cvtepi16_epi32(it);
        __m128i it4 = _mm_cvtepi16_epi32(it2);
        it = _mm_srli_si128(it, 8);
        it2 = _mm_srli_si128(it2, 8);
        it = _mm_cvtepi16_epi32(it);
        it2 = _mm_cvtepi16_epi32(it2);

        results = _mm_add_epi32(results, it);
        results2 = _mm_add_epi32(results2, it2);
        results = _mm_add_epi32(results, it3);
        results2 = _mm_add_epi32(results2, it4);
    }

    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    results2 = _mm_srli_si128(results, 4);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    return result;
}
size_t vec_i8x32n_sum(size_t size, const int8_t *src)
{
    return vec_i8x32n_sum_i32(size, src);
}

// returns i16x8
static __m128i vec_u8v32n_sum_i16x8(size_t size, const uint8_t *src)
{
    size_t units = size / 32;
    const __m128i *p = (const void*)src;

    const __m128i mask_lower = _mm_set1_epi16(0x00FF);
    __m128i results = _mm_setzero_si128();
    __m128i results2 = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i * 2];
        __m128i it2 = p[i * 2 + 1];

        __m128i it_hi = _mm_slli_epi16(it, 8);
        __m128i it2_hi = _mm_slli_epi16(it2, 8);
        __m128i it_lo = _mm_and_si128(it, mask_lower);
        __m128i it2_lo = _mm_and_si128(it2, mask_lower);

        results = _mm_adds_epi16(results, it_hi);
        results2 = _mm_adds_epi16(results2, it2_hi);
        results = _mm_adds_epi16(results, it_lo);
        results2 = _mm_adds_epi16(results2, it2_lo);
    }

    return _mm_adds_epi16(results, results2);
}
uint8_t vec_u8v32n_sum_u8(size_t size, const uint8_t *src)
{
    __m128i results = vec_u8v32n_sum_i16x8(size, src);
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 2));
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 4));
    results = _mm_adds_epi16(results, _mm_srli_si128(results, 8));
    size_t result = _mm_cvtsi128_si32(results) & 0xFFFF;

    return result > INT8_MAX ? INT8_MAX : result;
}
size_t vec_u8v32n_sum(size_t size, const uint8_t *src)
{
    const size_t unit_size = 0x4000 / (UINT8_MAX + 1) * 8;
    size_t units = size / unit_size;

    size_t result = 0;

    for (int i = 0; i < units; ++i)
    {
        __m128i results = vec_u8v32n_sum_i16x8(unit_size, src + i * unit_size);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_srli_si128(results, 16);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    size_t size2 = size % unit_size;
    size_t base = units * unit_size;

    if (size2 != 0)
    {
        __m128i results = vec_u8v32n_sum_i16x8(size2, src + base);
        const __m128i mask_lower = _mm_set1_epi32(0x0000FFFF);
        __m128i results2 = _mm_srli_si128(results, 16);
        results = _mm_and_si128(results, mask_lower);
        results = _mm_add_epi32(results, results2);
        results = _mm_add_epi32(results, _mm_srli_si128(results, 4));
        size_t result2 = _mm_cvtsi128_si32(results) & 0xFFFF;

        size_t result0 = result;
        result = result0 + result2;
        if (result < result0) result = SIZE_MAX;
    }

    return result;
}


/* deviation sum of square */

size_t vec_i32x8n_dss_with_avg(size_t size, const int32_t *src, int32_t _avg)
;
int64_t vec_i32x8n_dss_with_avg_i64(size_t size, const int32_t *src, int32_t _avg);
size_t vec_i16x16n_dss_with_avg(size_t size, const int16_t *src, int16_t _avg)
;
int32_t vec_i16x16n_dss_with_avg_i32(size_t size, const int16_t *src, int16_t _avg);
int64_t vec_i16x16n_dss_with_avg_i64(size_t size, const int16_t *src, int16_t _avg);
uint32_t vec_u16v16n_dss_with_avg_u32(size_t size, const uint16_t *src, uint16_t _avg);
uint64_t vec_u16v16n_dss_with_avg_u64(size_t size, const uint16_t *src, uint16_t _avg);
size_t vec_i8x32n_dss_with_avg(size_t size, const int8_t *src, int8_t _avg)
{
    return vec_i8x32n_dss_with_avg_i32(size, src, _avg);
}
int16_t vec_i8x32n_dss_with_avg_i16(size_t size, const int8_t *src, int8_t _avg)
{
    return vec_i8x32n_dss_with_avg_i32(size, src, _avg);
}
int32_t vec_i8x32n_dss_with_avg_i32(size_t size, const int8_t *src, int8_t _avg)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;

    __m128i results = _mm_setzero_si128();
    __m128i avgs = _mm_set1_epi16(_avg);

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        __m128i it2 = _mm_srli_si128(it, 8);

        it = _mm_cvtepi8_epi16(it);
        it2 = _mm_cvtepi8_epi16(it2);

        it = _mm_subs_epi16(it, avgs);
        it2 = _mm_subs_epi16(it2, avgs);

        __m128i it3 = _mm_cvtepi16_epi32(it);
        __m128i it4 = _mm_cvtepi16_epi32(it2);
        it = _mm_srli_si128(it, 8);
        it2 = _mm_srli_si128(it2, 8);
        it = _mm_cvtepi16_epi32(it);
        it2 = _mm_cvtepi16_epi32(it2);

        it = _mm_mullo_epi32(it, it);
        it2 = _mm_mullo_epi32(it2, it2);
        it3 = _mm_mullo_epi32(it3, it3);
        it4 = _mm_mullo_epi32(it4, it4);

        it = _mm_add_epi32(it, it3);
        it2 = _mm_add_epi32(it2, it4);

        it = _mm_add_epi32(it, it2);
        results = _mm_add_epi32(results, it);
    }

    __m128i results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    results = _mm_srli_si128(results, 4);
    int32_t result2 = _mm_cvtsi128_si32(results);
    return result + result2;
}
size_t vec_u8v32n_dss_with_avg(size_t size, const uint8_t *src, uint8_t _avg)
;
uint16_t vec_u8v32n_dss_with_avg_u16(size_t size, const uint8_t *src, uint8_t _avg);
uint32_t vec_u8v32n_dss_with_avg_u32(size_t size, const uint8_t *src, uint8_t _avg);

size_t vec_i32x8n_dss(size_t size, const int32_t *src)
;
// {
//     size_t _avg = vec_i32x8n_avg(size, src);
//     return vec_i32x8n_dss_with_avg(size, src, _avg);    
// }
int64_t vec_i32x8n_dss_i64(size_t size, const int32_t *src);
size_t vec_i16x16n_dss(size_t size, const int16_t *src)
;
int32_t vec_i16x16n_dss_i32(size_t size, const int16_t *src);
int64_t vec_i16x16n_dss_i64(size_t size, const int16_t *src);
uint32_t vec_u16v16n_dss_u32(size_t size, const uint16_t *src);
uint64_t vec_u16v16n_dss_u64(size_t size, const uint16_t *src);
size_t vec_i8x32n_dss(size_t size, const int8_t *src)
{
    size_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg(size, src, _avg);
}
int16_t vec_i8x32n_dss_i16(size_t size, const int8_t *src);
int32_t vec_i8x32n_dss_i32(size_t size, const int8_t *src);
size_t vec_u8v32n_dss(size_t size, const uint8_t *src)
;
uint16_t vec_u8v32n_dss_u16(size_t size, const uint8_t *src);
uint32_t vec_u8v32n_dss_u32(size_t size, const uint8_t *src);


/* residual sum of square */

size_t vec_i32x8n_rss(size_t size, const int32_t *src, const int32_t *predicted)
;
int64_t vec_i32x8n_rss_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_rss(size_t size, const int16_t *src, const int16_t *predicted)
;
int32_t vec_i16x16n_rss_i32(size_t size, const int16_t *src, const int16_t *predicted);
int64_t vec_i16x16n_rss_i64(size_t size, const int16_t *src, const int16_t *predicted);
uint32_t vec_u16v16n_rss_u32(size_t size, const uint16_t *src, const uint16_t *predicted);
uint64_t vec_u16v16n_rss_u64(size_t size, const uint16_t *src, const uint16_t *predicted);
size_t vec_i8x32n_rss(size_t size, const uint8_t *src, const uint8_t *predicted)
{
    return vec_i8x32n_rss_i32(size, src, predicted);
}
int16_t vec_i8x32n_rss_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_rss_i32(size_t size, const int8_t *src, const int8_t *predicted)
{
    size_t units = size / 16;
    const __m128i *p = (const void*)src;
    const __m128i *q = (const void*)predicted;

    __m128i results = _mm_setzero_si128();

    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];
        __m128i pred = q[i];

        __m128i it2 = _mm_srli_si128(it, 8);
        __m128i pred2 = _mm_srli_si128(pred, 8);
        it = _mm_cvtepi8_epi16(it);
        it2 = _mm_cvtepi8_epi16(it2);
        pred = _mm_cvtepi8_epi16(pred);
        pred2 = _mm_cvtepi8_epi16(pred2);

        it = _mm_subs_epi16(it, pred);
        it2 = _mm_subs_epi16(it2, pred2);

        __m128i it3 = _mm_cvtepi16_epi32(it);
        __m128i it4 = _mm_cvtepi16_epi32(it2);
        it = _mm_srli_si128(it, 8);
        it2 = _mm_srli_si128(it2, 8);
        it = _mm_cvtepi16_epi32(it);
        it2 = _mm_cvtepi16_epi32(it2);

        it = _mm_mullo_epi32(it, it);
        it2 = _mm_mullo_epi32(it2, it2);
        it3 = _mm_mullo_epi32(it3, it3);
        it4 = _mm_mullo_epi32(it4, it4);

        it = _mm_add_epi32(it, it3);
        it2 = _mm_add_epi32(it2, it4);

        it = _mm_add_epi32(it, it2);
        results = _mm_add_epi32(results, it);
    }

    __m128i results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi32(results, results2);

    int32_t result = _mm_cvtsi128_si32(results);
    results = _mm_srli_si128(results, 4);
    int32_t result2 = _mm_cvtsi128_si32(results);
    return result + result2;
}
uint16_t vec_u8v32n_rss_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_rss_u32(size_t size, const uint8_t *src, const uint8_t *predicted);

/* explained sum of square */
/* sigma{ (y[i] - avg(x))^2 } */

size_t vec_i32x8n_ess(size_t size, const int32_t *src, const int32_t *predicted)
;
int64_t vec_i32x8n_ess_i64(size_t size, const int32_t *src, const int32_t *predicted);
size_t vec_i16x16n_ess(size_t size, const int16_t *src, const int16_t *predicted)
;
int32_t vec_i16x16n_ess_i32(size_t size, const int16_t *src, const int16_t *predicted);
int64_t vec_i16x16n_ess_i64(size_t size, const int16_t *src, const int16_t *predicted);
uint32_t vec_u16v16n_ess_u32(size_t size, const uint16_t *src, const uint16_t *predicted);
uint64_t vec_u16v16n_ess_u64(size_t size, const uint16_t *src, const uint16_t *predicted);
size_t vec_i8x32n_ess(size_t size, const int8_t *src, const int8_t *predicted)
{
    size_t _avg = vec_i8x32n_avg(size, src);
    return vec_i8x32n_dss_with_avg(size, predicted, _avg);
}
int16_t vec_i8x32n_ess_i16(size_t size, const int8_t *src, const int8_t *predicted);
int32_t vec_i8x32n_ess_i32(size_t size, const int8_t *src, const int8_t *predicted);
size_t vec_u8v32n_ess(size_t size, const uint8_t *src, const uint8_t *predicted);
uint16_t vec_u8v32n_ess_u16(size_t size, const uint8_t *src, const uint8_t *predicted);
uint32_t vec_u8v32n_ess_u32(size_t size, const uint8_t *src, const uint8_t *predicted);


/* hisotgram */

void vec_i8x32n_get_histogram_i8x4(size_t size, const int8_t *src, int8_t min, int8_t max, uint8_t *out_bins)
{
    int8_t mins0[4];
    int8_t maxs0[4];
    {
        double w = (max - min) / 4;
        for (int i = 0; i < 4; ++i) mins0[i] = min + w * i;
        for (int i = 0; i < 3; ++i) maxs0[i] = mins0[i + 1] - 1;
        maxs0[4] = max;
    }

    size_t units = size / 16;
    const __m128i *p = (const void*)src;
    __m128i mins = _mm_set_epi8(
            mins0[0], mins0[1], mins0[2], mins0[3],  mins0[0], mins0[1], mins0[2], mins0[3],
            mins0[0], mins0[1], mins0[2], mins0[3],  mins0[0], mins0[1], mins0[2], mins0[3]);
    __m128i maxs = _mm_set_epi8(
            maxs0[0], maxs0[1], maxs0[2], maxs0[3],  maxs0[0], maxs0[1], maxs0[2], maxs0[3],
            maxs0[0], maxs0[1], maxs0[2], maxs0[3],  maxs0[0], maxs0[1], maxs0[2], maxs0[3]);
    __m128i results = _mm_setzero_si128();
    __m128i ones = _mm_set1_epi8(1);


    for (int i = 0; i < units; ++i)
    {
        __m128i it = p[i];

        __m128i flags = _mm_cmpgte_epi8(it, mins);
        __m128i flags2 = _mm_cmplte_epi8(it, maxs);
        flags = _mm_and_si128(flags, flags2);
        flags = _mm_and_si128(flags, ones);
        results _mm_add_epi8(results, flags);

        mins = _mm_and_si128(_mm_slli(mins, 1), _mm_srli(mins, 15));
        maxs = _mm_and_si128(_mm_slli(maxs, 1), _mm_srli(maxs, 15));
        
        flags = _mm_cmpgte_epi8(it, mins);
        flags2 = _mm_cmplte_epi8(it, maxs);
        flags = _mm_and_si128(flags, flags2);
        flags = _mm_and_si128(flags, ones);
        flags = _mm_and_si128(_mm_srli(maxs, 1), _mm_slli(maxs, 15));
        results _mm_add_epi8(results, flags);

        mins = _mm_and_si128(_mm_slli(mins, 1), _mm_srli(mins, 15));
        maxs = _mm_and_si128(_mm_slli(maxs, 1), _mm_srli(maxs, 15));
        
        flags = _mm_cmpgte_epi8(it, mins);
        flags2 = _mm_cmplte_epi8(it, maxs);
        flags = _mm_and_si128(flags, flags2);
        flags = _mm_and_si128(flags, ones);
        flags = _mm_and_si128(_mm_srli(maxs, 2), _mm_slli(maxs, 14));
        results _mm_add_epi8(results, flags);

        mins = _mm_and_si128(_mm_slli(mins, 1), _mm_srli(mins, 15));
        maxs = _mm_and_si128(_mm_slli(maxs, 1), _mm_srli(maxs, 15));
        
        flags = _mm_cmpgte_epi8(it, mins);
        flags2 = _mm_cmplte_epi8(it, maxs);
        flags = _mm_and_si128(flags, flags2);
        flags = _mm_and_si128(flags, ones);
        flags = _mm_and_si128(_mm_srli(maxs, 3), _mm_slli(maxs, 13));
        results _mm_add_epi8(results, flags);

        mins = _mm_and_si128(_mm_slli(mins, 1), _mm_srli(mins, 15));
        maxs = _mm_and_si128(_mm_slli(maxs, 1), _mm_srli(maxs, 15));
    }

    __m128i results2 = _mm_srli_si128(results, 8);
    results = _mm_add_epi8(results, results2);
    results2 = _mm_srli_si128(results, 4);
    results = _mm_add_epi8(results, results2);

    uint32_t results_u32 = _mm_cvtsi128_epi32();
    for (int i = 0; i < 4; ++i)
    {
        out_bins[i] = results_u32 & 0xFF;
        results_u32 >>= 8;
    }
}
void vec_i8x32n_get_histogram_u8x8(size_t size, const int8_t *src, int8_t min, int8_t max, uint8_t *out_bins);
void vec_i8x32n_get_histogram_u16x4(size_t size, const int8_t *src, int8_t min, int8_t max, uint16_t *out_bins);
void vec_i8x32n_get_histogram_u16x8(size_t size, const int8_t *src, int8_t min, int8_t max, uint16_t *out_bins);
void vec_i8x32n_get_histogram_u32x4(size_t size, const int8_t *src, int8_t min, int8_t max, uint32_t *out_bins);
