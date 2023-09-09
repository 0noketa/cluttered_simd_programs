#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#include "data16.inc"

int main()
{
    for (int i = 0; i < UINT16_MAX; ++i)
    {
        int16_t r0;
        int16_t r1;
        vec_i16v16n_get_minmax(DATA16_SIZE, data16, &r0, &r1);

        printf("%d .. %d\n", (int)r0, (int)r1);

    }

    return 0;
}
