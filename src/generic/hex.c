#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <ctype.h>



static uint32_t col2half(uint32_t c)
{
    return c - (isdigit(c) ? 48
        : isupper(c) ? 65 - 10
        : islower(c) ? 97 - 10
        : 0);
}


// input size: 64n bytes
// output size: 32n bytes, a half of input size
// size: encoded size in bytes, 64n bytes
int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst)
{
    // 8n bytes as uint32x8 x 2n per step
    size_t units = input_size / 2;

    // for (size_t i = 0; i < units; ++i)
    int i;
    #pragma omp parallel for num_threads(4)
    for (i = 0; i < units; ++i)
    {
        uint8_t c0 = src[i * 2];
        uint8_t c1 = src[i * 2 + 1];

        c0 = col2half(c0);
        c1 = col2half(c1);

        c0 = (c0 << 4) | c1;
 
        dst[i] = c0;
    }

    return 1;
}

