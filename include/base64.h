#ifndef _BASE64__H_
#define _BASE64__H_

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


int base64_encode(size_t size, const uint8_t *src, uint8_t *dst);
int base64_decode(size_t input_size, const uint8_t *src, uint8_t *dst, size_t *out_padding);


#ifdef __cplusplus
}
#endif

#endif
