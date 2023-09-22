#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/hex.h"

// #include "data_hex256.inc"
#include "data_hex524288.inc"

uint8_t dst[DATA_HEX_DECODED_SIZE] = {0,};

int main()
{

    // for (int i = 0; i < 1; ++i)
    for (int i = 0; i < UINT8_MAX * 4; ++i)
    {
        base16_128n_decode(DATA_HEX_SIZE, data_hex, dst);

        if (memcmp(dst, data_hex_decoded, DATA_HEX_DECODED_SIZE))
        {
            puts("error");

            int n = 256;
            for (int i = 0; i < DATA_HEX_DECODED_SIZE; ++i)
            {
                if (dst[i] == data_hex_decoded[i]) continue;

                printf("%06X: %02X %02X\n", (int)i, (int)dst[i], (int)data_hex_decoded[i]);

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
 * score = nearest(avg(samples*3), samples*3)
 * msvc_x86_cflags={ all: "/O2", autoomp2: "/Qpar" },
 * ispc_x86_cflags={ all: "-O3" },
 * gcc_x86_cflags={ all: "-Ofast" },
 * dpcpp_x64_xe_cflags={ all: "-Ofast -fopenmp -fiopenmp -fopenmp-targets=spir64_x86_64 -march=alderlake -mtune=alderlake" },


* 2023-09-08  lifebook(Mobile Intel Celeron (coppermine) 500MHz = {cores: 1, thrds: ?})  ddr1(100MHz)  cc=gcc  Linux(32bit)
x86           0m34.62s
mmx(consts/2)  0m9.81s   28.336%
* 2023-09-08  eeepc r11cx(Atom N2600 = {cores: 2, thrds: 4})  ddr3(1066MHz)  cc=gcc  Linux(32bit PAE)
x86            0m7.982s
x86+omp_x4     0m2.537s  31.784%
mmx            0m1.511s  18.930%
mmx(consts/2)  0m1.475s  18.479%
* 2023-09-08  dynabook(Core i5-4200M = {cores: 2, thrds: 4})  ddr3(1600MHz)  cc=msvc  Windows(64bit)
x86            0m6.763s
x86+omp_x4     0m2.699s  39.908%
mmx            0m0.461s   6.817%
ispc_sse2      0m0.774s  11.445%
avx2           0m0.185s   2.735%
avx2+omp_x4    0m1.050s  15.526%
ispc_avx2      0m0.524s   7.748%
* 2023-09-08  lifebook(Core i5-1235U = {cores: 10, thrds: 12})  ddr4(3200MHz)  cc=msvc  dpcpp=x64+num_threads(8)  Windows(64bit)
x86            0m5.168s
x86+omp_x4     0m1.674s  32.392%
x86+omp_x8     0m1.038s  20.085%
dpcpp+xe_x8    0m0.156s   3.019%
mmx            0m0.522s  10.101%
ispc_sse2      0m0.407s   7.875%
avx2           0m0.196s   3.792%
ispc_avx2      0m0.364s   7.043%

**/
