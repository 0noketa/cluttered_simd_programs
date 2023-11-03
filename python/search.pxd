# distutils: sources = ../src/search.c
# distutils: include_dirs = ../src/

include "cstdint.pxi"

cdef extern from "../include/search.h":
    cpdef int32_t vec_i32v8n_get_min(size_t size, const int32_t *src)
    cpdef int32_t vec_i32v8n_get_max(size_t size, const int32_t *src)
    cpdef void vec_i32v8n_get_minmax(size_t size, const int32_t *src, int32_t *out_min, int32_t *out_max)
    cpdef int16_t vec_i16v16n_get_min(size_t size, const int16_t *src)
    cpdef int16_t vec_i16v16n_get_max(size_t size, const int16_t *src)
    cpdef void vec_i16v16n_get_minmax(size_t size, const int16_t *src, int16_t *out_min, int16_t *out_max)
    cpdef int8_t vec_i8v32n_get_min(size_t size, const int8_t *src)
    cpdef int8_t vec_i8v32n_get_max(size_t size, const int8_t *src)
    cpdef void vec_i8v32n_get_minmax(size_t size, const int8_t *src, int8_t *out_min, int8_t *out_max)

    cpdef int32_t vec_i32v8n_count_i32(size_t size, const int32_t *src, int32_t value)
    cpdef size_t vec_i32v8n_count(size_t size, const int32_t *src, int32_t value)
    cpdef int16_t vec_i16v16n_count_i16(size_t size, const int16_t *src, int16_t value)
    cpdef size_t vec_i16v16n_count(size_t size, const int16_t *src, int16_t value)
    cpdef int8_t vec_i8v32n_count_i8(size_t size, const int8_t *src, int8_t value)
    cpdef size_t vec_i8v32n_count(size_t size, const int8_t *src, int8_t value)
