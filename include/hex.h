
#ifndef _HEX__H_
#define _HEX__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>


int base16_64n_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_64n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_4n_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_4n_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_encode_u(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_encode_l(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_inplace_encode_u(size_t input_size, uint8_t *data);
int base16_inplace_encode_l(size_t input_size, uint8_t *data);

int base16_128n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_32n_decode(size_t input_size, const uint32_t *src, uint32_t *dst);
int base16_8n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_2n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
int base16_inplace_decode(size_t input_size, uint8_t *data);


#endif
