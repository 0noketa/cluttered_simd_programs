// 8KB LUT for 12bit->16bit base64 encode
#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/base64.h"

#define TARGET_IS_LITTLE_ENDIAN


alignas(32) uint8_t src[8] = {0,};
alignas(32) uint8_t dst[8] = {0,};


int main(int argc, char *argv[])
{
    puts("static const uint_fast16_t base64_12bit_encode_[0x1000] = {");
    for (uint32_t i = 0; i < (1 << 12); ++i)
    {
        src[0] = 0;
        src[1] = (i >> 8) & 0x0F;
        src[2] = i & 0xFF;

        base64_encode(3, src, dst);

        char *s = i == (1 << 12) - 1 ? "\n"
                : i > 0 && i % 16 == 0 ? ",\n"
                : ", ";

#ifdef TARGET_IS_LITTLE_ENDIAN
        printf("0x%02X%02X%s", (int)dst[3], (int)dst[2], s);
#else
        printf("0x%02X%02X%s", (int)dst[2], (int)dst[3], s);
#endif
    }
    puts("};");

    return 0;
}
