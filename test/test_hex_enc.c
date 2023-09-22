#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <stdalign.h>

#include "../include/hex.h"

// #include "data_hex256.inc"
#include "data_hex524288.inc"

uint8_t dst[DATA_HEX_SIZE] = {0,};

int main()
{
    for (int i = 0; i < DATA_HEX_SIZE; ++i)
    {
        data_hex[i] = toupper(data_hex[i]);
    }

    // for (int i = 0; i < 1; ++i)
    for (int i = 0; i < UINT8_MAX * 4; ++i)
    {
        base16_64n_encode_u(DATA_HEX_DECODED_SIZE, data_hex_decoded, dst);

        if (memcmp(dst, data_hex, DATA_HEX_SIZE))
        {
            puts("error");

            int n = 256;
            for (int i = 0; i < DATA_HEX_SIZE / 4; ++i)
            {
                // if (dst[i] == data_hex[i]) continue;

                printf("%06X: %c(%02X) %c(%02X)\n",
                        (int)i,
                        (int)dst[i], (int)dst[i],
                        (int)data_hex[i], (int)data_hex[i]);

                if (!--n) break;
            }
        }
        else
        {
            puts("ok");
        }
    }

    return 0;
}


/* data_size=[0x80000], loop=UINT8_MAX*4, omp2_threads=4
 * score = nearest(avg(samples*5), samples*5)
 * msvc_x86_cflags={ all: "/O2", autoomp2: "/Qpar" },
 * ispc_x86_cflags={ all: "-O3" },
 * gcc_x86_cflags={ all: "-Ofast" },
 * gcc_arm_cflags={ all: "-Ofast" },
 * dpcpp_x64_xe_cflags={ all: "-Ofast -fopenmp -fiopenmp -fopenmp-targets=spir64_x86_64 -march=alderlake -mtune=alderlake" },

* 2023-09-20  armv6l(pi 1b) + gcc
generic         0m8.398s
armv6simd       0m8.091s  96.345%
* 2023-09-21  haswell(i5 4200M) + ddr3(1600MHz) + msvc
generic         0m0.461s
generic+omp_x4  0m0.363s  78.742%
avx2            0m0.157s  34.056%
avx2+omp_x4     0m0.215s

**/
