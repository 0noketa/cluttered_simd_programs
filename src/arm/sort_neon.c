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



void vec_i32v8n_reverse(size_t size, const int32_t *src, int32_t *dst)
{
	size_t units = size / 2;
	size_t units2 = units / 2;
	const int64x1_t *p = (const int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	for (int i = 0; i < units2; ++i)
	{
		uint64x1_t left = p[i * 2];
		uint64x1_t right = p[j * 2 + 1];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		left = vorr_u64(left_left, left_right);
		right = vorr_u64(right_left, right_right);

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[units2];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		it = vorr_u32(left, right);

		q[units2] = it;
	}
}
// current version is slow as generic version is.
void vec_i16v16n_reverse(size_t size, const int16_t *src, int16_t *dst)
{
	size_t units = size / 4;
	size_t units2 = units / 2;
	const int64x1_t *p = (const int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	for (int i = 0; i < units2; ++i)
	{
		uint64x1_t left = p[i * 2];
		uint64x1_t right = p[i * 2 + 1];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		uint32x2_t left32 = vreinterpret_u32_u64(vorr_u64(left_left, left_right));
		uint32x2_t right32 = vreinterpret_u32_u64(vorr_u64(right_left, right_right));

		uint32x2_t left_left32 = vshl_n_u32(left32, 16);
		uint32x2_t right_left32 = vshl_n_u32(right32, 16);
		uint32x2_t left_right32 = vshr_n_u32(left32, 16);
		uint32x2_t right_right32 = vshr_n_u32(right32, 16);

		left = vreinterpret_u64_u32(vorr_u32(left_left32, left_right32));
		right = vreinterpret_u64_u32(vorr_u32(right_left#2, right_right32));

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[units2];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		uint32x2_t it32 = vreinterpret_u32_u64(vorr_u64(left, right));

		uint32x2_t left32 = vshl_n_u32(it32, 16);
		uint32x2_t right32 = vshr_n_u32(it32, 16);

		it = vreinterpret_u64_u32(vorr_u32(left32, right32));

		q[units2] = it;
	}
}
void vec_i8v32n_reverse(size_t size, const int8_t *src, int8_t *dst)
{
	size_t units = size / 8;
	size_t units2 = units / 2;
	const int64x1_t *p = (const int64x1_t*)src;
	int64x1_t *q = (int64x1_t*)dst;

	for (int i = 0; i < units2; ++i)
	{
		uint64x1_t left = p[i * 2];
		uint64x1_t right = p[i * 2 + 1];

		uint64x1_t left_left = vshr_n_u64(left, 32);
		uint64x1_t right_left = vshr_n_u64(right, 32);
		uint64x1_t left_right = vshl_n_u64(left, 32);
		uint64x1_t right_right = vshl_n_u64(right, 32);

		uint32x2_t left32 = vreinterpret_u32_u64(vorr_u64(left_left, left_right));
		uint32x2_t right32 = vreinterpret_u32_u64(vorr_u64(right_left, right_right));

		uint32x2_t left_left32 = vshl_n_u32(left32, 16);
		uint32x2_t right_left32 = vshl_n_u32(right32, 16);
		uint32x2_t left_right32 = vshr_n_u32(left32, 16);
		uint32x2_t right_right32 = vshr_n_u32(right32, 16);

		uint16x4_t left16 = vreinterpret_u16_u32(vorr_u32(left_left32, left_right32));
		uint16x4_t right16 = vreinterpret_u16_u32(vorr_u32(right_left32, right_right32));

		uint16x4_t left_left16 = vshl_n_u16(left16, 8);
		uint16x4_t right_left16 = vshl_n_u16(right16, 8);
		uint16x4_t left_right16 = vshr_n_u16(left16, 8);
		uint16x4_t right_right16 = vshr_n_u16(right16, 8);

		left = vreinterpret_u64_u16(vorr_u16(left_left16, left_right16));
		right = vreinterpret_u64_u16(vorr_u16(right_left16, right_right16));

		q[units - 1 - i * 2 - 1] = right;
		q[units - 1 - i * 2] = left;
	}

	if (units & 1)
	{
		uint64x1_t it = p[units2];

		uint64x1_t left = vshr_n_u64(it, 32);
		uint64x1_t right = vshl_n_u64(it, 32);

		uint32x2_t it32 = vreinterpret_u32_u64(vorr_u64(left, right));

		uint32x2_t left32 = vshl_n_u32(it32, 16);
		uint32x2_t right32 = vshr_n_u32(it32, 16);

		uint16x4_t it16 = vreinterpret_u16_u32(vorr_u32(left32, right32));

		uint16x4_t left16 = vshl_n_u16(it16, 8);
		uint16x4_t right16 = vshr_n_u16(it16, 8);

		it = vreinterpret_u64_u16(vorr_u16(left16, right16));

		q[units2] = it;
	}
}




/* shift */

void vec_u256n_shl1(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 16;
    const __m128i *p = (const __m128i*)src;
    __m128i *q = (__m128i*)dst;

    __m128i mask_high = _mm_set1_epi8(UINT8_MAX - 1);
    __m128i mask_low = _mm_set1_epi8(1);
    __m128i mask_carry = _mm_set_epi32(0x01000000, 0, 0, 0);
    __m128i it = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        __m128i next = p[i + 1];
        __m128i shifted = _mm_slli_epi32(it, 1);
        __m128i shifted2 = _mm_srli_epi32(it, 7);
        shifted = _mm_and_si128(shifted, mask_high);
        shifted2 = _mm_and_si128(shifted2, mask_low);
        __m128i shifted3 = _mm_slli_si128(next, 15);
        shifted2 = _mm_srli_si128(shifted2, 1);
        shifted3 = _mm_srli_epi32(shifted3, 7);

        __m128i it2 = _mm_or_si128(shifted, shifted2);
        shifted3 = _mm_and_si128(shifted3, mask_carry);
        it2 = _mm_or_si128(it2, shifted3);

        q[i] = it2;

        it = next;
    }

    {
        __m128i shifted = _mm_slli_epi32(it, 1);
        __m128i shifted2 = _mm_srli_epi32(it, 7);
        shifted = _mm_and_si128(shifted, mask_high);
        shifted2 = _mm_and_si128(shifted2, mask_low);
        shifted2 = _mm_srli_si128(shifted2, 1);

        q[units - 1] = _mm_or_si128(shifted, shifted2);
    }
}
void vec_u256n_shl8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const uint64x1_t *p = (const uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        uint64x1_t it = p[i + 1];
        uint64x1_t carried = vshl_n_u64(it, 56);
        uint64x1_t shifted = vshr_n_u64(it0, 8);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = vshr_n_u64(it0, 8);
}
void vec_u256n_shr8(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const uint64x1_t *p = (const uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        uint64x1_t it = p[i - 1];
        uint64x1_t carried = vshr_n_u64(it, 56);
        uint64x1_t shifted = vshl_n_u64(it0, 8);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = vshl_n_u64(it0, 8);
}
void vec_u256n_shl32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const uint64x1_t *p = (const uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[0];

    for (size_t i = 0; i < units - 1; ++i)
    {
        uint64x1_t it = p[i + 1];
        uint64x1_t carried = vshl_n_u64(it, 32);
        uint64x1_t shifted = vshr_n_u64(it0, 32);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[size - 1] = vshr_n_u64(it0, 32);
}
void vec_u256n_shr32(size_t size, const uint8_t *src, uint8_t *dst)
{
    size_t units = size / 8;
    const uint64x1_t *p = (const uint64x1_t*)src;
    uint64x1_t *q = (uint64x1_t*)dst;

    uint64x1_t it0 = p[units - 1];

    for (size_t i = units - 1; i > 0; --i)
    {
        uint64x1_t it = p[i - 1];
        uint64x1_t carried = vshr_n_u64(it, 32);
        uint64x1_t shifted = vshl_n_u64(it0, 32);
        shifted = vorr_u64(shifted, carried);

        it0 = it;
        q[i] = shifted;
    }

    q[0] = vshl_n_u64(it0, 32);
}

void vec_u256n_rol8(size_t size, const uint8_t *src, uint8_t *dst)
{
    vec_u256n_shl8(size, src, dst);

    dst[size - 1] = src[0];
}
void vec_u256n_ror8(size_t size, const uint8_t *src, uint8_t *dst)
{
    vec_u256n_shr8(size, src, dst);

    dst[0] = src[size - 1];
}

void vec_u256n_rol32(size_t size, const uint8_t *src, uint8_t *dst)
;

void vec_u256n_ror32(size_t size, const uint8_t *src, uint8_t *dst)
;

