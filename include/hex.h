
#ifndef _HEX__H_
#define _HEX__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>


int base16_128n_encode(size_t input_size, const uint8_t *src, uint8_t *dst);

int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);


#endif
