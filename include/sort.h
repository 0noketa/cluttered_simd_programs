
#ifndef _SIMD_TOOLS__SORT__H_
#define _SIMD_TOOLS__SORT__H_

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>
#include <stdlib.h>


/* reverse */

extern void vec_i32v8n_inplace_reverse(size_t size, int32_t *data)
;
extern void vec_i32v4n_inplace_reverse(size_t size, int32_t *data)
;
extern void vec_i32v2n_inplace_reverse(size_t size, int32_t *data)
;
extern void vec_i16v16n_inplace_reverse(size_t size, int16_t *data)
;
extern void vec_i16v8n_inplace_reverse(size_t size, int16_t *data)
;
extern void vec_i16v4n_inplace_reverse(size_t size, int16_t *data)
;
extern void vec_i8v32n_inplace_reverse(size_t size, int8_t *data)
;
extern void vec_i8v16n_inplace_reverse(size_t size, int8_t *data)
;
extern void vec_i8v8n_inplace_reverse(size_t size, int8_t *data)
;

extern void vec_i32v8n_reverse(size_t size, const int32_t *src, int32_t *dst)
;
extern void vec_i32v4n_reverse(size_t size, const int32_t *src, int32_t *dst)
;
extern void vec_i32v2n_reverse(size_t size, const int32_t *src, int32_t *dst)
;
extern void vec_i16v16n_reverse(size_t size, const int16_t *src, int16_t *dst)
;
extern void vec_i16v8n_reverse(size_t size, const int16_t *src, int16_t *dst)
;
extern void vec_i16v4n_reverse(size_t size, const int16_t *src, int16_t *dst)
;
extern void vec_i8v32n_reverse(size_t size, const int8_t *src, int8_t *dst)
;
extern void vec_i8v16n_reverse(size_t size, const int8_t *src, int8_t *dst)
;
extern void vec_i8v8n_reverse(size_t size, const int8_t *src, int8_t *dst)
;
extern void bits256n_reverse(size_t size, const uint8_t *src, uint8_t *dst)
;


/* shift */

extern void bits256n_shl1(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_shr1(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_shl8(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_shl32(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_shr8(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_shr32(size_t size, const uint8_t *src, uint8_t *dst)
;

extern void bits256n_rol1(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_rol8(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_rol32(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_ror1(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_ror8(size_t size, const uint8_t *src, uint8_t *dst)
;
extern void bits256n_ror32(size_t size, const uint8_t *src, uint8_t *dst)
;

extern void bits256n_rol(size_t size, const uint8_t *src, int n, uint8_t *dst)
;
extern void bits256n_ror(size_t size, const uint8_t *src, int n, uint8_t *dst)
;


/* ascendant/descendant */

extern void  vec_i32v8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end)
;
// dst should be i32?
extern void  vec_i16v16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
;
// dst should be i16?
extern void  vec_i8v32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end)
;

extern int  vec_i32v8n_is_sorted_a(size_t size, const int32_t *src)
;
extern int  vec_i32v8n_is_sorted_d(size_t size, const int32_t *src)
;
extern int  vec_i32v8n_is_sorted(size_t size, const int32_t *src)
;
extern int  vec_i16v16n_is_sorted_a(size_t size, const int16_t *src)
;
extern int  vec_i16v16n_is_sorted_d(size_t size, const int16_t *src);
extern int  vec_i16v16n_is_sorted(size_t size, const int16_t *src)
;
extern int  vec_i8v32n_is_sorted_a(size_t size, const int8_t *src)
;
extern int  vec_i8v32n_is_sorted_d(size_t size, const int8_t *src)
;
extern int  vec_i8v32n_is_sorted(size_t size, const int8_t *src)
;




#endif
