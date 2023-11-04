#ifndef _BASE64__H_
#define _BASE64__H_

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


// ignores "input_size % 3" bytes
extern int base64_3n_encode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 12" bytes
extern int base64_12n_encode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 24" bytes
extern int base64_24n_encode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 48" bytes
extern int base64_48n_encode(size_t input_size, const uint8_t *src, uint8_t *dst);

// ignores "input_size % 4" bytes
extern int base64_4n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 16" bytes
extern int base64_16n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 32" bytes
extern int base64_32n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);
// ignores "input_size % 64" bytes
extern int base64_64n_decode(size_t input_size, const uint8_t *src, uint8_t *dst);

extern int base64_encode(size_t input_size, const uint8_t *src, uint8_t *dst);
extern int base64_decode(size_t input_size, const uint8_t *src, uint8_t *dst, size_t *out_padding);


#ifdef __cplusplus
}
#endif

#endif
