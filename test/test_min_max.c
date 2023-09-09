#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#include "data16.inc"

int main()
{
    for (int i = 0; i < UINT16_MAX; ++i)
    {
        int16_t r0 = vec_i16v16n_get_min(DATA16_SIZE, data16);
        int16_t r1 = vec_i16v16n_get_max(DATA16_SIZE, data16);

        printf("%d .. %d\n", (int)r0, (int)r1);

    }

    return 0;
}


/* data_size=[65536], loop=UINT16_MAX,
 * cflags={ generic: "-Ofast", simd: "-Os" }, sample=avg(tried3times)
2022-12-25  r11cx(Atom N2600)
    generic_min_max    0m34.306s
    generic_minmax     0m19.708s       57.45%
    mmx_min_max        0m11.916s       34.74%
    mmx_minmax         0m8.060s        23.50%
    sse2_min_max       0m6.105s        17.80%
    sse2_minmax        0m4.044s        11.79%
2022-12-26  dynabook(Core i5-4200M)
    generic_min_max    0m10.349s
    generic_minmax     0m3.079s        29.76%
    mmx_min_max        0m3.411s        32.96%
    mmx_minmax         0m2.194s        21.21%
    sse2_min_max       0m0.865s
    sse2_minmax        0m0.567s
    avx2_min_max       0m0.613s
    avx2_minmax        0m0.402s         3.89%
2023-01-06  dynabook(Core i5-4200M)
    generic_min_max    0m10.314s
    generic_minmax     0m3.226s
    mmx_min_max        0m3.416s
    mmx_minmax         0m2.197s        21.31%
    (ver andnot)       0m2.534s        24.57%
    sse2_min_max       0m0.861s
    sse2_minmax        0m0.563s
    avx2_min_max       0m0.607s
    avx2_minmax        0m0.410s         3.98%
2022-12-27  pi 1b(armv6l)
    generic_min_max    2m47.364s
    generic_minmax     1m36.372s
2023-01-01  pi 2b(Cortex-A7) (at 600MHz)
    generic_min_max    1m14.074s
    generic_minmax     0m58.451s
    neon_min_max       0m19.664s
    neon_minmax        0m11.687s       15.78%
2022-12-27  pi 3b(Cortex-A53)
    generic_min_max    0m36.159s
    generic_minmax     0m25.286s
    neon_min_max       0m10.266s
    neon_minmax        0m5.562s        15.39%
**/
