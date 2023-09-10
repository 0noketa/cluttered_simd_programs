
// ispc --addressing=32 --arch=x86 -O3 --emit-obj -h hex.h -o hex.o hex.ispc --target=avx2-i32x8
#define target_size_t uint32


bool isupper(uint32 c)
{
    return and(c >= 65, c <= 90);
}
bool islower(uint32 c)
{
    return and(c >= 97, c <= 122);
}
bool isalpha(uint32 c)
{
    return isupper(c) | islower(c);
}
bool isdigit(uint32 c)
{
    return and(c >= 48, c <= 57);
}

uint32 col2half(uint32 c)
{
#ifdef INCREMENTAL_PARSE
    c -= 48;
    if (c < 10) return c;
    c -= (65 - 48);
    if (c < 6) return c + 10;
    c -= (97 - 65);
    if (c < 6) return c + 10;
    return 0;
#else
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
#endif
}


// input size: 64n bytes
// output size: 32n bytes, a half of input size
// size: encoded size in bytes, 64n bytes
export uniform int base16_128n_decode(uniform target_size_t input_size, uniform const uint32 src[], uniform uint32 dst[])
{
    // 8n bytes as uint32x8 x 2n per step
    uniform target_size_t units = input_size / 4 / TARGET_WIDTH;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uint32 unit = src[i * TARGET_WIDTH + programIndex];

        uint32 c0 = (unit >> 0) & 0xFF;
        uint32 c1 = (unit >> 8) & 0xFF;
        uint32 c2 = (unit >> 16) & 0xFF;
        uint32 c3 = (unit >> 24) & 0xFF;

        c0 = col2half(c0);
        c1 = col2half(c1);
        c2 = col2half(c2);
        c3 = col2half(c3);

        c0 = (c0 << 4) | c1;
        c2 = (c2 << 4) | c3;

        // makes: 0x0000FFFF, 0xFFFF0000,  0x0000FFFF, 0xFFFF0000, ...
        if (programIndex % 2 == 0)
        {
            unit = (c0 << 0) | (c2 << 8);
        }
        else
        {
            unit = (c0 << 16) | (c2 << 24);
        }

        // packs [0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0]
        // into [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0]
        uniform uint32 uni_unit[TARGET_WIDTH];
        uni_unit[programIndex] = unit;

        if (programIndex % 2 == 0)
        {
            uni_unit[programIndex / 2] = unit | uni_unit[programIndex + 1];
        }

        if (programIndex < TARGET_WIDTH / 2)
        {
            dst[i * TARGET_WIDTH / 2 + programIndex] = uni_unit[programIndex];
        }
    }

    return 1;
}

