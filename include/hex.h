
#ifndef _SIMD_TOOLS__H_
#define _SIMD_TOOLS__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>


int base16_64n_decode(size_t input_size, uint8_t *dst, const uint8_t *src);


#endif
