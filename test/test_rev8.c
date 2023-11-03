#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#define DATA_SIZE 0x8000000

alignas(32) int8_t src[DATA_SIZE];
alignas(32) int8_t dst[DATA_SIZE];


void dump(int8_t *data, size_t idx)
{
    size_t idx2 = idx & ~31;
    for (int i = idx2; i < idx2 + 32; ++i)
    {
        printf("%s%4d,", (i == idx ? "*" : " "), (int)data[i]);
    }
    puts("");
}

int main()
{
    for (int i = 0; i < DATA_SIZE; ++i) src[i] = i;


    // for (int i = 0; i < UINT8_MAX/4; ++i)
    for (int i = 0; i < 1; ++i)
    {
        vec_i8v32n_reverse(DATA_SIZE, src, dst);

		for (int j = 0; j < DATA_SIZE; ++j)
		{
            if (src[j] != dst[DATA_SIZE - 1 - j]
                || src[DATA_SIZE - 1 - j] != dst[j])
            {
    			printf("error(reverse) at %d<->%d: %d<->%d != %d<->%d\n",
                    (int)j, (int)(DATA_SIZE - 1 - j),
                    (int)src[j], (int)src[DATA_SIZE - 1 - j],
                    (int)dst[j], (int)dst[DATA_SIZE - 1 - j]);

                dump(src, j);
                dump(dst, j);
                return 1;
            }
		}

        vec_i8v32n_inplace_reverse(DATA_SIZE, dst);

		for (int j = 0; j < DATA_SIZE; ++j)
		{
            if (src[j] != dst[j]
                || src[DATA_SIZE - 1 - j] != dst[DATA_SIZE - 1 - j])
            {
    			printf("error(inplace_reverse) at %d<->%d: %d<->%d != %d<->%d\n",
                    (int)j, (int)(DATA_SIZE - 1 - j),
                    (int)src[j], (int)src[DATA_SIZE - 1 - j],
                    (int)dst[j], (int)dst[DATA_SIZE - 1 - j]);

                dump(src, j);
                dump(dst, j);
                return 1;
            }
		}
    }

    return 0;
}


/* data_size=[0x8000000], loop=1,
 * cflags={ gcc: "-Ofast", msvc: "/O2" }, sample=avg(5samples)

* 2023-11-04  Core-i5-4200M  DDR3(1600MHz)  msvc(/O2)  x86
generic 0m0.736s
rot     0m0.643s
rot+stk 0m0.659s
omp_x4  0m0.664s
sse2    0m0.637s
avx2    0m0.632s
**/
