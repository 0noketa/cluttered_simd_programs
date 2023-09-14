#include <immintrin.h>

#if defined(__GNUC__)
#include <x86intrin.h>
#elif defined(_MSC_VER) 
#include <intrin.h>
#endif

#if sizeof(__m128i) != 16 || sizeof(__m256i) != 32
#error "unexpected padding (align <=16 is required"
#endif


// 384 bits = loads 3 times = 4 units

// A-a-0-+/
// 0..:'A'=65, 26..:'a'=97, 52..:'0'=48, 62:'+'=43, 63:'/'=47

// input pattern
// 8
// 11111100 00001111 11000000
// 16
// 11111100 00001111
// 11000000 11111100
// 00001111 11000000
// 32 (using this)
// 11111100 00001111 11000000 11111100
// 00001111 11000000 11111100 00001111
// 11000000 11111100 00001111 11000000

// // src in reg128(H .. L)(line = 64 bit):
//   ________ ________ ________ ________  EEFFFFFF DDDDEEEE CCCCCCDD AABBBBBB
//   9999AAAA 88888899 66777777 55556666  44444455 22333333 11112222 00000011
// // src2 (with preprocess (shr 64 as i128
//   EEFFFFFF DDDDEEEE CCCCCCDD AABBBBBB  9999AAAA 88888899 66777777 55556666
//   44444455 22333333 11112222 00000011  ________ ________ ________ ________
// // dst in reg128(H..L):
//   __FFFFFF __EEEEEE __DDDDDD __CCCCCC  __BBBBBB __AAAAAA __999999 __888888
//   __777777 __666666 __555555 __444444  __333333 __222222 __111111 __000000
// // src3 (offset: 16):
//   ________ ________ EEFFFFFF DDDDEEEE  CCCCCCDD AABBBBBB 9999AAAA 88888899 
//   66777777 55556666 44444455 22333333  11112222 00000011 ________ ________
// // can be shl in epi16:
//   ________ ________ ________ ________  ________ ________ ________ ________
//   ________ ________ ________ 5555____  ________ ________ ________ ______11
// // can be shl in epi32:
//   ________ ________ ________ ________  ________ ________ ________ AABBBBBB
//   ________ ________ 66777777 ____6666  ________ __333333 ____2222 ________
// // can be shl in epi64:
//   ________ ________ ________ ________  EEFFFFFF DDDDEEEE CCCCCCDD ________
//   ________ ________ ________ ________  44444455 ________ ________ ________
// // can be shr in epi8:
//   ________ ________ ________ ________  ________ ________ ________ ________
//   ________ ________ ________ ________  ________ 22______ 1111____ 000000__
// // requires 128bit shift
//   ________ ________ ________ ________  ________ ________ ________ ________
//   9999AAAA 88888899 ________ ________  ________ ________ ________ ________
// // can be shl in src2.epi16
//   ________ ____EEEE ______DD ________  ________ ________ ________ ________
//   ________ ________ ________ ________  ________ ________ ________ ________
// // can be shr in src2.epi16
//   EE______ ________ CCCCCC__ ________  ____AAAA ________ ________ ________
//   ________ ________ ________ ________  ________ ________ ________ ________
// // can be shr in src2.epi32
//   ________ DDDD____ CCCCCC__ ________  9999____ 88888899 ________ ________
//   44444455 ________ ________ ________  ________ ________ ________ ________

// // src3 (offset: 16):
//   ________ ________ EEFFFFFF DDDDEEEE  CCCCCCDD AABBBBBB 9999AAAA 88888899 
//   66777777 55556666 44444455 22333333  11112222 00000011 ________ ________
// // dst in reg128(H..L):
//   __FFFFFF __EEEEEE __DDDDDD __CCCCCC  __BBBBBB __AAAAAA __999999 __888888
//   __777777 __666666 __555555 __444444  __333333 __222222 __111111 __000000
// // shl epi8:
//   ________ ________ ________ ________  ________ ________ ________ ________
//   __777777 ____6666 ______55 ________  ________ ________ ________ ________
// // shl epi16:
//   ________ ________ ________ DDDD____  ________ __BBBBBB ________ ______99
//   ________ ________ ______## ________  ________ ________ ________ ________
// // shl epi32:
//   ________ ________ EEFFFFFF ####EEEE  ________ __###### ____AAAA ______##
//   ________ ________ ______## ________  ________ ________ ________ ________
// // shl epi64:
//   ________ ________ ######## ########  CCCCCCDD __###### ________ ______##
//   ________ ________ ______## ________  ________ ________ ________ ________
// // shr epi8:
//   ________ ________ ######## ########  ######## AA###### 9999#### 888888##
//   ____#### ____#### ______## ________  ________ ________ ________ ________
// // shr epi16:
//   ________ ________ ######## ########  ######## ######## ######## ########
//   6666#### ____#### 444444## ________  ____2222 ________ ________ ________
// // shr epi32:
//   ________ ________ ######## ########  ######## ######## ######## ########
//   ######## 5555#### ######## ________  1111#### 00000011 ________ ________
// // shr epi64:
//   ________ ________ ######## ########  ######## ######## ######## ########
//   ######## ######## ######## 22333333  ######## ######## ________ ________
// 64
// 11111100 00001111 11000000 11111100 00001111 11000000 11111100 00001111
// 11000000 11111100 00001111 11000000 11111100 00001111 11000000 11111100
// 00001111 11000000 11111100 00001111 11000000 11111100 00001111 11000000

static __m256i enc_chunk96(__m256i src);

// 128 * 3 -> 96 * 4
static inline void repack_3to4(
    __m256i *dst0, __m256i *dst1, __m256i *dst2, __m256i *dst3,
    __m128i src0, __m128i src1, __m128i src2);

static inline __m128i load_be_u32x4(void *src)
{
    uint32_t *p = src;

    union {
        __m128i packed;
        uint32_t arr[4];
    } tmp;

    tmp.arr[0] = _load_be_u32(p + 0);
    tmp.arr[1] = _load_be_u32(p + 1);
    tmp.arr[2] = _load_be_u32(p + 2);
    tmp.arr[3] = _load_be_u32(p + 3);

    return tmp.packed;
}

static inline __m256i load_be_u32x8(void *src)
{
    uint32_t *p = src;

    union {
        __m256i packed;
        uint32_t arr[8];
    } tmp;

    tmp.arr[0] = _load_be_u32(p + 0);
    tmp.arr[1] = _load_be_u32(p + 1);
    tmp.arr[2] = _load_be_u32(p + 2);
    tmp.arr[3] = _load_be_u32(p + 3);
    tmp.arr[4] = _load_be_u32(p + 4);
    tmp.arr[5] = _load_be_u32(p + 5);
    tmp.arr[6] = _load_be_u32(p + 6);
    tmp.arr[7] = _load_be_u32(p + 7);

    return tmp.packed;
}

int base64_48n_encode(size_t size, const uint8_t *src, uint8_t *dst)
{
    if (src == NULL || dst == NULL) return 0;

    size_t units = len / 48;

    __m128i *p = (void*) src;
    __m128i *q = (void*) dst;

    for (size_t i = 0; i < units; ++i)
    {
        __m128i src0 = load_be_u32x4(p + i * 48);
        __m128i src1 = load_be_u32x4(p + i * 48 + 1);
        __m128i src2 = load_be_u32x4(p + i * 48 + 2);

        // a:
        // 00000011 11112222 22333333 44444455 55556666 66777777 
        // 88888899 9999aaaa aabbbbbb ccccccdd ddddeeee eeffffff
        // b:
        // 00111111 22222233 33334444 44555555 66666677 77778888
        // 88999999 aaaa____ bbbbcccc ccdddddd eeeeeeff ffff0000
        // 00111111 22222233 33334444 4455____ 66666677 77778888
        // 88999999 aaaaaabb bbbbcccc ccdddddd eeeeeeff ffff____
        // 00111111 22222233 33334444 44555555 66666677 77778888
        // 88999999 aaaa____ bbbbcccc ccdddddd eeeeeeff ffff0000
        // 00111111 22222233 33334444 4455____ 66666677 77778888
        // 88999999 aaaaaabb bbbbcccc ccdddddd eeeeeeff ffff____
        // c: for 5 & a


        // f _ c b|_ 8 7 _|4 3 _ 0|f _ c b|_ 8 7 _|4 3 _ 0
        __m128i bit_shifter_0a = _mm_set_epi8(0,0,2,0, 0,2,0,0, 2,0,0,2,    0,0,2,0);
        __m128i bit_shifter_1a = _mm_set_epi8(0,2,0,0, 2,0,0,2,    0,0,2,0, 0,2,0,0);
        __m128i bit_shifter_2a = _mm_set_epi8(2,0,0,2,    0,0,2,0, 0,2,0,0, 2,0,0,2);
        // _ e d _|a 9 _ 6|5 _ 2 1|_ e d _|a 9 _ 6|5 _ 2 1
        __m128i bit_shifter_0b = _mm_set_epi8(0,2,0,0, 2,0,0,2, 0,0,2,0,    0,2,0,0);
        __m128i bit_shifter_1b = _mm_set_epi8(2,0,0,2, 0,0,2,0,    0,2,0,0, 2,0,0,2);
        __m128i bit_shifter_2b = _mm_set_epi8(0,0,2,0,    0,2,0,0, 2,0,0,2, 0,0,2,0);

        __m128i src0b = _mm_slli_epi64(src0, 4);
        __m128i src1b = _mm_slli_epi64(src1, 4);
        __m128i src2b = _mm_slli_epi64(src2, 4);

        src0 = _mm_srl_epi8(src0, bit_shifter_0a);
        src1 = _mm_srl_epi8(src1, bit_shifter_1a);
        src2 = _mm_srl_epi8(src2, bit_shifter_2a);
        src0b = _mm_srl_epi8(src0b, bit_shifter_0b);
        src1b = _mm_srl_epi8(src1b, bit_shifter_1b);
        src2b = _mm_srl_epi8(src2b, bit_shifter_2b);

        _mm_store(q++, dst0);
        _mm_store(q++, dst1);
        _mm_store(q++, dst2);
        _mm_store(q++, dst3);
    }

    if (rest == 0)
        return;

    size_t offset = units * BASE64_BLOCK_SIZE;
    size_t rest2 = rest % 3;
    size_t chunks = rest / 3;
    uint32_t buf = 0;

    for (size_t i = 0; i < chunks; ++i)
    {
        // no SIMD
    }
}

__m128i enc_chunk96(__m128i src)
{
    return src;
}


static inline void repack_3to4(
    __m128i *dst0, __m128i *dst1, __m128i *dst2, __m128i *dst3,
    __m128i src0, __m128i src1, __m128i src2)
{
    static const __m128i mask0 = { .m128_i32 = {
            UINT32_MAX, UINT32_MAX, UINT32_MAX, 0}};
    static const __m128i mask1 = { .m128_i32 = {
            UINT32_MAX, UINT32_MAX, 0, 0}};
    static const __m128i mask2 = { .m128_i32 = {
            UINT32_MAX, 0, 0, 0}};

    // 3210, 7654, ba98 (col of col: int32) -> _210, _543, _876, _ba9

    __m256i v0;
    v0 = _mm256_inserti128(v0, src0, 0);
    __m256i v1;
    v1 = _mm256_inserti128(v1, src1, 0);
    __m256i v2;
    v2 = _mm256_inserti128(v2, src2, 0);

    // 3210, _ -> _210, _
    __m256i v0_r = _mm256_bsrli_epi128(v0, 4);
    v0_r = _mm256_bslli_epi128(v0_r, 4);
    // 3210, _ -> ___3, _
    __m256i v0_l = _mm256_bsrli_epi128(v0, 12);

    // 7654, _ -> __76, _
    __m256i v1_r = _mm256_bsrli_epi128(v1, 8);
    // 7654, _ -> _54_, _
    __m256i v1_l = _mm256_bslli_epi128(v1, 8);
    v1_l = _mm256_bsrli_epi128(v1_l, 4);

    // ba98, _ -> _ba9, _
    __m256i v2_r = _mm256_bsrli_epi128(v2, 4);
    // ba98, _ -> _8__, _
    __m256i v2_l = _mm256_bslli_epi128(v2, 12);
    v2_l = _mm256_bsrli_epi128(v2_l, 4);

    //_765
    __m128i tmp0 = _mm256_extracti128_si256(v0_r, 0);
    //_543
    __m128i tmp1_0 = _mm256_extracti128_si256(v0_l, 0);
    __m128i tmp1_1 = _mm256_extracti128_si256(v1_l, 0);
    __m128i tmp1 = _mm256_and_si128(tmp1_0, tmp1_1);
    //_876
    __m128i tmp2_0 = _mm256_extracti128_si256(v1_r, 0);
    __m128i tmp2_1 = _mm256_extracti128_si256(v2_l, 0);
    __m128i tmp2 = _mm256_and_si128(tmp2_0, tmp2_1);
    //_ba9
    __m128i tmp3 = _mm256_extracti128_si256(v2_l, 0);

    // redistribute into codes above
    // _abc -> abc_
    __m256i tv0;
    tv0 = _mm256_inseti128(tv0, tmp0, 0);
    tv0 = _mm256_inseti128(tv0, tmp1, 1);
    __m256i tv1;
    tv1 = _mm256_inseti128(tv1, tmp2, 0);
    tv1 = _mm256_inseti128(tv1, tmp3, 1);

    tv0 = _mm256_bslli_epi128(tv0, 4);
    tv1 = _mm256_bslli_epi128(tv1, 4);

    tmp0 = _mm256_extracti128_si256(tv0, 0);
    tmp1 = _mm256_extracti128_si256(tv0, 1);
    tmp2 = _mm256_extracti128_si256(tv1, 0);
    tmp3 = _mm256_extracti128_si256(tv1, 1);

    *dst0 = tmp0;
    *dst1 = tmp1;
    *dst2 = tmp2;
    *dst3 = tmp3;
}
