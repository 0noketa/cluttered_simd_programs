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


#ifdef BASE64_INT64
static int base64_srcbufsize(int size, int *out_chnks)
{
	int chnks = size / 3;
	int rem = size % 3;

	if (out_chnks)
		*out_chnks = chnks;

	return chnks * 3 + rem + 1;
}
int base64_any2_enc(uint8_t *dst, const uint8_t *src, size_t size)
{
	int rem, chnks;
	int offsetDst, offsetSrc;
	int ch;
	register uint64_t q0, q1, q2, q3;

	chnks = size / 6;
	rem = size - chnks * 6;

	offsetSrc = 0;
	offsetDst = 0;

	for (size_t i = 0; i < chnks; ++i)
	{
		q0 = *(uint16_t*)(src + offsetSrc + 0);
		q1 = *(uint16_t*)(src + offsetSrc + 2);
		q2 = *(uint16_t*)(src + offsetSrc + 4);
/*
6 2, 4 4,  2 6, 6 2,  4 4, 2 6
6, 2 4, 4 2, 6, 6, 2 4, 4 2, 6

big:
++++++++--------
>>1111110000000000<<<<<<
>>>>0000001111110000<<<<
>>>>>>0000000000001111<<
++++++++--------++++++++--------++++++++
>>>>>>>>>>>>>>>>>>>>>>1100000000000000<<<<<<<<<<
>>>>>>>>>>>>>>>>>>>>>>>>0011111100000000<<<<<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>0000000011111100<<<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000000000000011<<<<
++++++++--------++++++++--------++++++++--------++++++++
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1111000000000000<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000111111000000<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000000000111111
++++++++--------++++++++--------++++++++--------++++++++--------
little:
++++++++--------++++++++--------++++++++--------++++++++--------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>111111
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000001111110000<<<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0000000000001111<<<<<<<<<<<<<<<<<<
++++++++--------++++++++--------++++++++--------++++++++--------
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1100000000000000<<
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0011111100000000<<<<<<<<<<<<<<<<
>>>>>>>>>>>>>>>>>>0000000011111100<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
>>>>0000000000000011<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
++++++++--------++++++++--------++++++++--------++++++++
>>>>>>>>>>>>>>>>>>>>1111000000000000<<<<
>>>>>>0000111111000000<<<<<<<<<<<<<<<<<<
00111111<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
++++++++--------++++++++--------
*/
#ifdef BASE64_BIG_ENDIAN
		q3 =
			((q0 & 0xFC00) << 46) |
			((q0 & 0x03F0) << 44) |
			((q0 & 0x000F) << 42) |
			((q1 & 0xC000) << 26) |
			((q1 & 0x3F00) << 24) |
			((q1 & 0x00FC) << 22) |
			((q1 & 0x0003) << 20) |
			((q2 & 0xF000) << 4) |
			((q2 & 0x0FC0) << 2) |
			((q2 & 0x003F) << 0);
#else
		q3 =
			((q0 & 0xFC00) >> 10) |
			((q0 & 0x03F0) << 4) |
			((q0 & 0x000F) << 18) |
			((q1 & 0xC000) << 2) |
			((q1 & 0x3F00) << 16) |
			((q1 & 0x00FC) << 30) |
			((q1 & 0x0003) << 44) |
			((q2 & 0xF000) << 28) |
			((q2 & 0x0FC0) << 42) |
			((q2 & 0x003F) << 56);
#endif
		*(int64_t*)(dst + offsetDst + 0) = q3;


		offsetSrc += 6;
		offsetDst += 8;
	}

	for (size_t i = 0; i < chnks * 8; ++i)
	{
		ch = dst[offsetDst + i];

		dst[offsetDst + i] =
			  ch < 26 ? ch + 'A'
			: ch < 52 ? ch + 'a' - 52
			: ch < 62 ? ch + '0' - 62
			: ch == 62 ? '+'
			: '/';
	}

	return 1;
}
#endif

#ifndef BASE64_GENERIC_ENC
int base64_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = size / 3;
    size_t rem = size % 3;

    const uint8_t *p = src;
    uint8_t *q = dst;

    for (size_t i = 0; i < units; ++i)
    {
        uint32_t buf = *p++ << 16;
        buf |= *p++ << 8;
        buf |= *p++;

        *q++ = cs[(buf >> 18) & 0x3F];
        *q++ = cs[(buf >> 12) & 0x3F];
        *q++ = cs[(buf >> 6) & 0x3F];
        *q++ = cs[(buf >> 0) & 0x3F];
    }

    // 1 11111111 -> 111111 11____ _ _
    // 2 11111111 22222222 -> 111111 112222 2222__ _
    if (rem == 1)
    {
        uint32_t x = *p;

        *q++ = cs[(x >> 2) & 0x3F];
        *q++ = cs[((x & 3) << 4) & 0x3F];
        *q++ = '=';
        *q = '=';
    }
    else if (rem == 2)
    {
        uint32_t x = *p++;
        uint32_t y = *p;

        *q++ = cs[(x >> 2) & 0x3F];
        *q++ = cs[(((x & 3) << 4) | (y >> 4)) & 0x3F];
        *q++ = cs[((y & 0x0F) << 2) & 0x3F];
        *q = '=';
    }

    return 1;
}
#endif




int base64_decode(size_t input_size, const uint8_t *src, uint8_t *dst, size_t *out_rem)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = input_size / 4;
    units -= (units > 0 && input_size % 4 == 0);
    const uint8_t *p = src;
    uint8_t *q = dst;

    uint32_t buf = 0;

    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        int c0 = src[i * 4 + 0];
        int c1 = src[i * 4 + 1];
        int c2 = src[i * 4 + 2];
        int c3 = src[i * 4 + 3];
        int x0 = cs2[c0];
        int x1 = cs2[c1];
        int x2 = cs2[c2];
        int x3 = cs2[c3];

        buf = (x0 << 18) | (x1 << 12) | (x2 << 6) | (x3 << 0);

        dst[i * 3 + 0] = (buf >> 16) & 0xFF;
        dst[i * 3 + 1] = (buf >> 8) & 0xFF;
        dst[i * 3 + 2] = (buf >> 0) & 0xFF;
    }

    size_t result_rem = 0;
    if (units * 4 < input_size)
    {
        buf = 0;
        size_t i;
        int rem = 0;
        for (i = units * 4; i < input_size; ++i)
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

