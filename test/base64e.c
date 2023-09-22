#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/base64.h"


#define BLOCK_SIZE 0xC0000

alignas(32) uint8_t encoded[BLOCK_SIZE * 4 / 3] = {0,};
alignas(32) uint8_t decoded[BLOCK_SIZE] = {0,};


int main(int argc, char *argv[])
{
    FILE *fIn = (argc > 1) ? fopen(argv[1], "rb") : stdin;

    if (fIn == NULL) return 1;

    while (!feof(fIn))
    {
        size_t input_size = fread(decoded, 1, BLOCK_SIZE, fIn);
        if (input_size == 0) continue;

        size_t rem;
        base64_48n_encode(input_size, decoded, encoded);

        size_t output_size = input_size * 4 / 3;

        fwrite(encoded, 1, output_size, stdout);
    }

    if (fIn != stdin) fclose(fIn);

    return 0;
}


/* input_file="ispc-v1.21.0-windows.zip", input_size=44078388, processed_size=44078352, omp2_threads=4
 * score = nearest(avg(samples*5), samples*5)
 * msvc_x86_cflags={ all: "/O2", autoomp2: "/Qpar" },
 * ispc_x86_cflags={ all: "-O3" },
 * gcc_x86_cflags={ all: "-Ofast" },
 * gcc_arm_cflags={ all: "-Ofast" },
 * dpcpp_x64_xe_cflags={ all: "-Ofast -fopenmp -fiopenmp -fopenmp-targets=spir64_x86_64 -march=alderlake -mtune=alderlake" },

* 2023-09-21  i5-4200M + ddr3(1600MHz) + msvc
generic         0m0.164s
generic+omp_x4  0m0.196s
lut             0m0.151s
lut+movbe       0m0.154s
lut2            0m0.140s  85.366%
mmx             0m0.371s
avx2            0m0.179s
ispc_sse2x8     0m0.312s
ispc_avx2       0m0.263s
**/