typedef unsigned int8 uint8;
typedef unsigned int32 uint32;

// inline uint8 base64_encc(uint8 n)
// {
//     n &= 0x3F;

//     return n
//         + (n < 26 ? 0x41
//             : n < 52 ? 0x61 - 26
//             : n < 62 ? 0x30 - 52
//             : n == 62 ? 0x2B - 62
//             : 0x2F - 63);
// }

// output will be the same code
inline uint8 base64_encc(uint8 n)
{
    n &= 0x3F;

    if (n < 26)
        return n + 0x41;
    else if (n < 52)
        return n - 26 + 0x61;
    else if (n < 62)
        return n - 52 + 0x30;
    else if (n == 62)
        return 0x2B;
    else
        return 0x2F;
}

uint8 base64v_encc(uint8 n)
{
    n &= 0x3F;

    if (n == 0)
        return 0x2B;
    else if (n == 1)
        return 0x2F;
    else if (n < 12)
        return n + 0x30 - 2;
    else if (n >= 38)
        return n + 0x61 - 38;
    else
        return n + 0x41 - 12;
}

uniform uint32 enc24_load(uniform uint8 *src)
{
    return reduce_add((src[0] << 16) | (src[1] << 8) | src[2]);
}
void enc24_store(uint8 *dst, uint32 buf)
{
    for (int i = 0; i < 4; ++i)
    {
        dst[i] = base64_encc((buf >> (i * 6)) & 0xFF);
    }
}
void enc24(uint8 *dst, uint8 *src)
{
    uniform uint32 buf = reduce_add((src[0] << 16) | (src[1] << 8) | src[2]);

    for (int i = 0; i < 4; ++i)
    {
        dst[i] = reduce_add(base64_encc((buf >> (i * 6)) & 0xFF));
    }
    // dst[1] = base64_encc((buf >> 12) & 0xFF);
    // dst[2] = base64_encc((buf >> 6) & 0xFF);
    // dst[3] = base64_encc((buf >> 0) & 0xFF);
}
export void base64_enc24(uniform uint8 dst[4], uniform uint8 src[3])
{
    enc24(dst, src);
}

export void base64_enc48(uniform uint8 dst[64], uniform uint8 src0[48])
{
    foreach (i = 0 ... 16)
    {
        uint8 *src = src0 + i * 3;
        uniform uint32 buf = enc24_load(src);

        enc24_store(dst + (i * 4), buf);
    }
}

void v_enc12(uniform uint8 * uniform dst, uniform uint8 * uniform src)
{
    foreach (i = 0 ... 3, j = 0 ... 4)
    {
        uint8 *p = src + i;
        uint32 buf = (p[0] << 16) | (p[4] << 8) | p[8];

        dst[j] = base64v_encc(buf >> (i * j) & 0xFF);
    }
}
export void base64v_enc24(uniform uint8 dst[32], uniform uint8 src[24])
{
    v_enc12(dst, src);
    v_enc12(dst + 16, src + 12);
}
export void base64v_enc48(uniform uint8 dst[64], uniform uint8 src[48])
{
    v_enc12(dst, src);
    v_enc12(dst + 16, src + 12);
    v_enc12(dst + 32, src + 24);
    v_enc12(dst + 48, src + 36);
}
