#define target_size_t uint32


inline uint32 encode_byte(uint32 n)
{
    if (n < 26)
        return n + 65;
    else if (n < 52)
        return n - 26 + 97;
    else if (n < 62)
        return n - 52 + 48;
    else if (n == 62)
        return 43;
    else
        return 47;
}
uniform inline uint32 encode_byte(uniform uint32 n)
{
    if (n < 26)
        return n + 65;
    else if (n < 52)
        return n - 26 + 97;
    else if (n < 62)
        return n - 52 + 48;
    else if (n == 62)
        return 43;
    else
        return 47;
}

inline uint32 decode_byte(uint32 n)
{
    n -= 43;
    if (n == 0) return 62;

    n -= 4;
    if (n == 0) return 63;

    n -= 1;
    if (n < 10) return n + 52;

    n -= 8;
    if (n < 26) return n;

    n -= 30;
    if (n < 26) return n + 26;

    return 0;
}
uniform inline uint32 decode_byte(uniform uint32 n)
{
    n -= 43;
    if (n == 0) return 62;

    n -= 4;
    if (n == 0) return 63;

    n -= 1;
    if (n < 10) return n + 52;

    n -= 8;
    if (n < 26) return n;

    n -= 30;
    if (n < 26) return n + 26;

    return 0;
}


export uniform int base64_24n_encode(uniform target_size_t input_size, uniform uint32 src[], uniform uint32 dst[])
{
    if (src == NULL || dst == NULL) return 0;

    uniform target_size_t units = input_size / 24;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t base = i * 6;
        uniform target_size_t dst_base = i * 8;

        uniform uint32 cs0[8];
        uniform uint32 cs1[8];
        uniform uint32 cs2[8];
        uniform uint32 cs3[8];

        for (uniform int j = 0; j < 2; ++j)
        {
            uniform uint32 unit0 = src[base + j * 3];
            uniform uint32 unit1 = src[base + j * 3 + 1];
            uniform uint32 unit2 = src[base + j * 3 + 2];

            // reg32s
            //        in_0                                  in_1                                  in_2
            // out_0: [         22333333 11112222 00000011] [                                   ] [                                   ]
            // out_1: [44444455                           ] [                  66777777 55556666] [                                   ]
            // out_2: [                                   ] [9999AAAA 88888899                  ] [                           AABBBBBB]
            // out_3: [                                   ] [                                   ] [EEFFFFFF DDDDEEEE CCCCCCDD         ]
            cs0[j * 4 + 0] = ((unit0 & 0x000000FC) >>  2);
            cs1[j * 4 + 0] = ((unit0 & 0x00000003) <<  4) | ((unit0 & 0x0000F000) >> 12);
            cs2[j * 4 + 0] = ((unit0 & 0x00C00000) >> 22) | ((unit0 & 0x00000F00) >>  6);
            cs3[j * 4 + 0] = ((unit0 & 0x003F0000) >> 16);

            cs0[j * 4 + 1] = ( unit0/*&0xFC000000*/>> 26);
            cs1[j * 4 + 1] = ((unit0 & 0x03000000) >> 20) | ((unit1 & 0x000000F0) >>  4);
            cs2[j * 4 + 1] = ((unit1 & 0x0000C000) >> 14) | ((unit1 & 0x0000000F) <<  2);
            cs3[j * 4 + 1] = ((unit1 & 0x00003F00) >>  8);

            cs0[j * 4 + 2] = ((unit1 & 0x00FC0000) >> 18);
            cs1[j * 4 + 2] = ((unit1 & 0x00030000) >> 12) | ( unit1/*&0xF0000000*/>> 28);
            cs2[j * 4 + 2] = ((unit2 & 0x000000C0) >>  6) | ((unit1 & 0x0F000000) >> 22);
            cs3[j * 4 + 2] = ((unit2 & 0x0000003F) >>  0);

            cs0[j * 4 + 3] = ((unit2 & 0x0000FC00) >> 10);
            cs1[j * 4 + 3] = ((unit2 & 0x00000300) >>  4) | ((unit2 & 0x00F00000) >> 20);
            cs2[j * 4 + 3] = ((unit2 & 0x000F0000) >> 14) | ( unit2/*&0xC0000000*/>> 30);
            cs3[j * 4 + 3] = ((unit2 & 0x3F000000) >> 24);
        }


#if TARGET_ELEMENT_WIDTH == 4 && TARGET_WIDTH == 4
        for (uniform target_size_t j = 0; j < 2; ++j)
        {
            uint32 c0 = encode_byte(cs0[j * 4 + programIndex]);
            uint32 c1 = encode_byte(cs1[j * 4 + programIndex]);
            uint32 c2 = encode_byte(cs2[j * 4 + programIndex]);
            uint32 c3 = encode_byte(cs3[j * 4 + programIndex]);

            uint32 dst_unit = (c0 << 0) | (c1 << 8) | (c2 << 16) | (c3 << 24);

            dst[dst_base + j * 4 + programIndex] = dst_unit;
        }
#elif TARGET_ELEMENT_WIDTH == 4 && TARGET_WIDTH == 8
        for (uniform target_size_t j = 0; j < 1; ++j)
        {
            uint32 c0 = encode_byte(cs0[programIndex]);
            uint32 c1 = encode_byte(cs1[programIndex]);
            uint32 c2 = encode_byte(cs2[programIndex]);
            uint32 c3 = encode_byte(cs3[programIndex]);

            uint32 dst_unit = (c0 << 0) | (c1 << 8) | (c2 << 16) | (c3 << 24);

            dst[dst_base + programIndex] = dst_unit;
        }
#else
#error "uses u32x4 or u32x8"
#endif
    }

    return 1;
}
