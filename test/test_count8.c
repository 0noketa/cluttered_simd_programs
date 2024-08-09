#include <stdio.h>
#include <stdint.h>
#include <stdalign.h>

#include "../include/simd_tools.h"

#include "data8.inc"


int main()
{
    for (int i = 0; i < /*32*/ DATA8_SIZE; ++i)
    {
        size_t result0 = 0;
        size_t it = data8[i];
        for (size_t j = 0; j < DATA8_SIZE; ++j)
        {
            result0 += (data8[j] == it);
        }

        size_t result = vec_i8x32n_count(DATA8_SIZE, data8, it);

        if (result != result0)
		{
			printf("#%d:%d   %d:u32, %d:i16\n", i, (int)it,  (int)result0, (int)result);
		}
    }

    return 0;
}
