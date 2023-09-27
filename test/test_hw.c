#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#include "data8.inc"

int main()
{
    // for (int i = 0; i < 1; ++i)
    for (int i = 0; i < DATA8_SIZE; ++i)
    {
        size_t n = vec_u256n_get_hamming_weight(DATA8_SIZE, data8);

        printf("%u\n", (unsigned)n);
    }

    return 0;
}


/*
 * data_size=65536, loop=65536

* 2023-09-27  coppermine(Mobile Intel Celeron 500Mhz)  DDR1(100MHz)  gcc(-Ofast -no-pie)
generic    0m 43.41s
lut        0m 23.02s
mmx128     0m 18.23s
mmx64      0m 22.03s
mmx_man128 0m 19.46s
mmx_man64  0m 23.74s
* 2023-09-27  Atom-N2600  DDR3(800MHz)  gcc(-Ofast -no-pie)  x86
generic    0m12.508s
generic_x4 0m5.875s
lut        0m16.183s
lut_x4     0m5.340s
mmx128     0m5.766s
mmx64      0m7.763s
mmx_man128 0m6.221s
mmx_man64  0m8.764s
sse2       0m3.060s
* 2023-09-27  Core-i5-4200M  DDR3(1600MHz)  msvc(/O2)  x64
* x86
generic    0m2.779s
generic_x4 0m1.668s
lut        0m1.118s
lut_x4     0m0.873s
mmx128     0m2.137s
mmx64      0m2.279s
mmx_man128 0m2.330s
sse2       0m1.059s
avx2       0m0.502s
popcnt     0m1.179s
* x64
generic    0m3.005s
generic_x4 0m1.794s
lut        0m1.282s
lut_x4     0m0.823s
sse2       0m0.971s
avx2       0m0.648s
popcnt     0m0.680s

**/
