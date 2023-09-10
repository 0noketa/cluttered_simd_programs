#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

#define load_8x8_from_bytes(dst, p) \
    dst = vset_lane_u8((p)[0], dst, 0); \
    dst = vset_lane_u8((p)[1], dst, 1); \
    dst = vset_lane_u8((p)[2], dst, 2); \
    dst = vset_lane_u8((p)[3], dst, 3); \
    dst = vset_lane_u8((p)[4], dst, 4); \
    dst = vset_lane_u8((p)[5], dst, 5); \
    dst = vset_lane_u8((p)[6], dst, 6); \
    dst = vset_lane_u8((p)[7], dst, 7);
#define load_8x16_from_bytes(dst, p) \
    dst = vsetq_lane_u8((p)[0], dst, 0); \
    dst = vsetq_lane_u8((p)[1], dst, 1); \
    dst = vsetq_lane_u8((p)[2], dst, 2); \
    dst = vsetq_lane_u8((p)[3], dst, 3); \
    dst = vsetq_lane_u8((p)[4], dst, 4); \
    dst = vsetq_lane_u8((p)[5], dst, 5); \
    dst = vsetq_lane_u8((p)[6], dst, 6); \
    dst = vsetq_lane_u8((p)[7], dst, 7); \
    dst = vsetq_lane_u8((p)[8], dst, 8); \
    dst = vsetq_lane_u8((p)[9], dst, 9); \
    dst = vsetq_lane_u8((p)[10], dst, 10); \
    dst = vsetq_lane_u8((p)[11], dst, 11); \
    dst = vsetq_lane_u8((p)[12], dst, 12); \
    dst = vsetq_lane_u8((p)[13], dst, 13); \
    dst = vsetq_lane_u8((p)[14], dst, 14); \
    dst = vsetq_lane_u8((p)[15], dst, 15);

#define load_bytes_from_8x8(p, src) \
    (p)[0] = vget_lane_u8(src, 0); \
    (p)[1] = vget_lane_u8(src, 1); \
    (p)[2] = vget_lane_u8(src, 2); \
    (p)[3] = vget_lane_u8(src, 3); \
    (p)[4] = vget_lane_u8(src, 4); \
    (p)[5] = vget_lane_u8(src, 5); \
    (p)[6] = vget_lane_u8(src, 6); \
    (p)[7] = vget_lane_u8(src, 7);
#define load_bytes_from_16x4(p, src) \
    (p)[0] = vget_lane_u16(src, 0); \
    (p)[1] = vget_lane_u16(src, 1); \
    (p)[2] = vget_lane_u16(src, 2); \
    (p)[3] = vget_lane_u16(src, 3);
#define load_bytes_from_8x16(p, src) \
    (p)[0] = vgetq_lane_u8(src, 0); \
    (p)[1] = vgetq_lane_u8(src, 1); \
    (p)[2] = vgetq_lane_u8(src, 2); \
    (p)[3] = vgetq_lane_u8(src, 3); \
    (p)[4] = vgetq_lane_u8(src, 4); \
    (p)[5] = vgetq_lane_u8(src, 5); \
    (p)[6] = vgetq_lane_u8(src, 6); \
    (p)[7] = vgetq_lane_u8(src, 7); \
    (p)[8] = vgetq_lane_u8(src, 8); \
    (p)[9] = vgetq_lane_u8(src, 9); \
    (p)[10] = vgetq_lane_u8(src, 10); \
    (p)[11] = vgetq_lane_u8(src, 11); \
    (p)[12] = vgetq_lane_u8(src, 12); \
    (p)[13] = vgetq_lane_u8(src, 13); \
    (p)[14] = vgetq_lane_u8(src, 14); \
    (p)[15] = vgetq_lane_u8(src, 15);

void dump_u8x8(const char *pfx, uint8x8_t src) {
    uint8_t dst[8] = {0,};
    load_bytes_from_8x8(dst, src);

    if (pfx) fputs(pfx, stdout);
    for (int i = 0; i < 8; ++i)
        printf("%02X ", (int)dst[i]);
    if (pfx) puts("");
}
void dump_u8x16(const char *pfx, uint8x16_t src) {
    uint8_t dst[16] = {0,};
    load_bytes_from_8x16(dst, src);

    if (pfx) fputs(pfx, stdout);
    for (int i = 0; i < 16; ++i)
        printf("%02X ", (int)dst[i]);
    if (pfx) puts("");
}

void dump_u16x4(const char *pfx, uint16x4_t src) {
    uint16_t dst[4] = {0,};
    load_bytes_from_16x4(dst, src);

    if (pfx) fputs(pfx, stdout);
    for (int i = 0; i < 4; ++i)
        printf("%04X ", (int)dst[i]);
    if (pfx) puts("");
}



#define BLOCKS_LEN 32

// #ifdef 
#define def_const_packed_u8x8(name, v) \
    static const uint8x8_t name = { \
        (v), (v), (v), (v),  (v), (v), (v), (v) \
    };
#define def_const_packed_u8x16(name, v) \
    static const uint8x16_t name = { \
        (v), (v), (v), (v),  (v), (v), (v), (v), \
        (v), (v), (v), (v),  (v), (v), (v), (v) \
    };

#define def_const_packed_u16x4(name, v) \
    static const uint16x4_t name = { \
        (v), (v), (v), (v) \
    };

#define def_const_packed_u16x8(name, v) \
    static const uint16x8_t name = { \
        (v), (v), (v), (v),  (v), (v), (v), (v) \
    };
// #endif

def_const_packed_u8x8(diff_num, '0')
def_const_packed_u8x8(diff_upper, 'A' - '0')
def_const_packed_u8x8(diff_lower, 'a' - 'A')
def_const_packed_u8x8(packed_8x8_6, 6)
def_const_packed_u8x8(packed_8x8_8, 8)
def_const_packed_u8x8(packed_8x8_10, 10)
def_const_packed_u16x4(packed_16x4_255, 255)
def_const_packed_u16x4(packed_16x4_65280, 65280)
def_const_packed_u8x16(diff16_num, '0')
def_const_packed_u8x16(diff16_upper, 'A' - '0')
def_const_packed_u8x16(diff16_lower, 'a' - 'A')
def_const_packed_u8x16(packed_8x16_6, 6)
def_const_packed_u8x16(packed_8x16_8, 8)
def_const_packed_u8x16(packed_8x16_10, 10)
def_const_packed_u16x8(packed_16x8_255, 255)
def_const_packed_u16x8(packed_16x8_65280, 65280)
def_const_packed_u16x8(packed_16x8_0, 0)


static inline uint16x4_t b16_copy_decoded_block_0(uint8x8_t src) {
    src = vsub_u8(src, diff_num);
    uint8x8_t dst = vclt_u8(src, packed_8x8_10);
    dst = vand_u8(src, dst);

    src = vsub_u8(src, diff_upper);
    uint8x8_t tmp = vclt_u8(src, packed_8x8_6);
    uint8x8_t tmp2 = vadd_u8(src, packed_8x8_10);
    tmp = vand_u8(tmp, tmp2);
    dst = vorr_u8(dst, tmp);

    src = vsub_u8(src, diff_lower);
    tmp = vclt_u8(src, packed_8x8_6);
    tmp2 = vadd_u8(src, packed_8x8_10);
    tmp = vand_u8(tmp, tmp2);
    dst = vorr_u8(dst, tmp);

    uint16x4_t dst16 = vreinterpret_u16_u8(dst);
    uint16x4_t dst16_hi = vand_u16(dst16, packed_16x4_65280);
    dst16 = vand_u16(dst16, packed_16x4_255);
    dst16 = vshl_n_u16(dst16, 12);

    return vadd_u16(dst16, dst16_hi);
}
static inline uint8x8_t b16_copy_decoded_block(uint8x8_t src, uint8x8_t src2) {
    uint16x4_t dst_hi = b16_copy_decoded_block_0(src);
    uint64x1_t dst64_hi = vreinterpret_u64_u16(dst_hi);
    uint16x4_t dst_lo = b16_copy_decoded_block_0(src2);
    uint64x1_t dst64_lo = vreinterpret_u64_u16(dst_lo);
    uint64x2_t dst64 = {0,};
    dst64 = vsetq_lane_u64(vget_lane_u64(dst64_hi, 0), dst64, 0);
    dst64 = vsetq_lane_u64(vget_lane_u64(dst64_lo, 0), dst64, 1);
    uint16x8_t dst16 = vreinterpretq_u16_u64(dst64);
    uint16x8_t empty16 = {0,};

    return vaddhn_u16(dst16, empty16);
    /*
    src2 = src - '0';
    num = src2 < 10 ? src2 : 0;
    src3 = src2 - ('A' - '0');
    upper = src3 < 6 ? src3 + 10 : 0;
    src4 = src2 - ('a' - 'A');
    lower = src4 < 6 ? src4 + 10 : 0;
    */
}
static inline uint8x8_t b16_copy_decoded_block_2(uint8x16_t src) {
    src = vsubq_u8(src, diff16_num);
    uint8x16_t dst = vcltq_u8(src, packed_8x16_10);
    dst = vandq_u8(src, dst);

    src = vsubq_u8(src, diff16_upper);
    uint8x16_t tmp = vcltq_u8(src, packed_8x16_6);
    uint8x16_t tmp2 = vaddq_u8(src, packed_8x16_10);
    tmp = vandq_u8(tmp, tmp2);
    dst = vorrq_u8(dst, tmp);

    src = vsubq_u8(src, diff16_lower);
    tmp = vcltq_u8(src, packed_8x16_6);
    tmp2 = vaddq_u8(src, packed_8x16_10);
    tmp = vandq_u8(tmp, tmp2);
    dst = vorrq_u8(dst, tmp);

    uint16x8_t dst16 = vreinterpretq_u16_u8(dst);
    uint16x8_t dst16_hi = vandq_u16(dst16, packed_16x8_65280);
    dst16 = vandq_u16(dst16, packed_16x8_255);
    dst16 = vshlq_n_u16(dst16, 12);

    dst16 = vaddq_u16(dst16, dst16_hi);
    uint16x8_t empty16 = {0,};

    return vaddhn_u16(dst16, empty16);
}


// input size: 64n bytes
// output size: 32n bytes, a half of input size
// size: encoded size in bytes, 64n bytes
int base16_128n_decode(size_t input_size, const uint32_t *src, uint32_t *dst)
{
    // 8n bytes as uint32x8 x 2n per step
    size_t units = input_size / 2 / 4;

    // for (size_t i = 0; i < units; ++i)
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
#ifdef USE_NEON_8x16
        uint8x16_t src1 = {0,};
        load_8x16_from_bytes(src1, p);

        p += 16;

        uint8x8_t dst8x8 = b16_copy_decoded_block_2(src1);

        load_bytes_from_8x8((dst + i * 8), dst8x8);
#else
        uint8x8_t src1 = {0,};
        load_8x8_from_bytes(src1, p);

        uint8x8_t src2 = {0,};
        load_8x8_from_bytes(src2, p + 8);
        p += 16;
        ++i;

        uint8x8_t dst8x8 = b16_copy_decoded_block(src1, src2);

        load_bytes_from_8x8((dst + i * 8), dst8x8);
#endif
    }

    return 1;
}

