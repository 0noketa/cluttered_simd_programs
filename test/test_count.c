#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#include "data16.inc"


int main()
{
    for (int i = 0; i < /*UINT8_MAX * 4*/ DATA16_SIZE; ++i)
    {
        size_t result0 = 0;
        size_t it = data16[i];
        for (size_t j = 0; j < DATA16_SIZE; ++j)
        {
            result0 += (data16[j] == it);
        }

        size_t result = vec_i16v16n_count(DATA16_SIZE, data16, it);

        if (result != result0)
		{
			printf("#%d:%d   %d:u32, %d:i16\n", i, (int)it,  (int)result0, (int)result);
		}
    }

    return 0;
}
