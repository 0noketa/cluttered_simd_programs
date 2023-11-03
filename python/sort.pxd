# distutils: sources = ../src/sort.c
# distutils: include_dirs = ../src/


include "cstdint.pxi"

cdef extern from "../include/sort.h":
    cpdef void vec_i32v8n_inplace_reverse(size_t size, int32_t *data)
    cpdef void vec_i16v16n_inplace_reverse(size_t size, int16_t *data)
    cpdef void vec_i8v32n_inplace_reverse(size_t size, int8_t *data)

    cpdef void vec_i32v8n_reverse(size_t size, const int32_t *src, int32_t *dst)
    cpdef void vec_i16v16n_reverse(size_t size, const int16_t *src, int16_t *dst)
    cpdef void vec_i8v32n_reverse(size_t size, const int8_t *src, int8_t *dst)

    cpdef void vec_i32v8n_get_sorted_index(size_t size, const int32_t *src, int32_t element, int32_t *out_start, int32_t *out_end)
    cpdef void vec_i16v16n_get_sorted_index(size_t size, const int16_t *src, int16_t element, int16_t *out_start, int16_t *out_end)
    cpdef void vec_i8v32n_get_sorted_index(size_t size, const int8_t *src, int8_t element, int8_t *out_start, int8_t *out_end)

    cpdef int vec_i32v8n_is_sorted_a(size_t size, const int32_t *src)
    cpdef int vec_i32v8n_is_sorted_d(size_t size, const int32_t *src)
    cpdef int vec_i32v8n_is_sorted(size_t size, const int32_t *src)
    cpdef int vec_i16v16n_is_sorted_a(size_t size, const int16_t *src)
    cpdef int vec_i16v16n_is_sorted_d(size_t size, const int16_t *src)
    cpdef int vec_i16v16n_is_sorted(size_t size, const int16_t *src)
    cpdef int vec_i8v32n_is_sorted_a(size_t size, const int8_t *src)
    cpdef int vec_i8v32n_is_sorted_d(size_t size, const int8_t *src)
    cpdef int vec_i8v32n_is_sorted(size_t size, const int8_t *src)
