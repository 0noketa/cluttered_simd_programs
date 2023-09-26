// 66KB LUT for 16bit->8bit popcnt
#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"


alignas(32) uint8_t src[32] = {0,};


int main(int argc, char *argv[])
{
    puts("static const uint_fast8_t hw_16bit_lut_[0x10000] = {");
    for (uint32_t i = 0; i < (1 << 16); ++i)
    {
        src[0] = (i >> 8) & 0xFF;
        src[1] = i & 0xFF;

        int r = vec_u256n_get_hamming_weight(32, src);

        char *s = i == (1 << 16) - 1 ? "\n"
                : i > 0 && i % 32 == 0 ? ",\n"
                : ",";

        printf("%2d%s", r, s);
    }
    puts("};");

    return 0;
}
