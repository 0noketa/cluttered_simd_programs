#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/search.h"

#include "data16.inc"

int main()
{
    for (int i = 0; i < 1; ++i)
    // for (int i = 0; i < INT16_MAX; ++i)
    {
        int16_t min0 = INT16_MAX;
        size_t min0_idx = SIZE_MAX;
        int16_t max0 = INT16_MIN;
        size_t max0_idx = SIZE_MAX;
        for (int j = 0; j < DATA16_SIZE; ++j)
        {
            int16_t it = data16[j];
            if (it < min0)
            {
                min0 = it;
                min0_idx = j;
            }
            if (it > max0)
            {
                max0 = it;
                max0_idx = j;
            }
        }

        size_t min_idx = vec_i16v16n_get_min_index(DATA16_SIZE, data16);
        int16_t min = vec_i16v16n_get_min(DATA16_SIZE, data16);
        size_t max_idx = vec_i16v16n_get_max_index(DATA16_SIZE, data16);
        int16_t max = vec_i16v16n_get_max(DATA16_SIZE, data16);

        if (min0 != min || min0_idx != min_idx
                || max0 != max || max0_idx != max_idx)
        {
            printf("expected: %d @ %d .. %d @ %d\n",
                (int)min0, (int)min0_idx,
                (int)max0, (int)max0_idx);

            printf("results : %d @ %d .. %d @ %d\n",
                (int)min, (int)min_idx,
                (int)max, (int)max_idx);
        }

        break;
    }

    return 0;
}


/* data_size=[65536], loop=UINT16_MAX, prog=[min, max]|[minmax],
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
