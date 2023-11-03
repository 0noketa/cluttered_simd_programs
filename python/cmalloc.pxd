
"""
#if defined(_WIN64) || defined(_WIN32)
"""
cdef extern from "<malloc.h>":
    cpdef void* _aligned_malloc(size_t size, size_t align)
    cpdef void _aligned_free(void *p)
"""
#else
"""
# cdef extern from "<stdlib.h>":
#     cpdef void* aligned_alloc(size_t size, size_t align)
#     cpdef void free(void *p)

# cdef inline void* _aligned_malloc(size_t size, size_t align):
#     return aligned_alloc(size, align)
# cdef inline void _aligned_free(void *p):
#     free(p)
"""
#endif
"""
