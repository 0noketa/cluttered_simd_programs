DEF INT8_MAX = 0x7F
DEF UINT8_MAX = 0xFF
DEF INT16_MAX = 0x7FFF
DEF UINT16_MAX = 0xFFFF
DEF INT32_MAX = 0x7FFFFFFF
DEF UINT32_MAX = 0xFFFFFFFF
DEF INT64_MAX = 0x7FFFFFFFFFFFFFFF
DEF UINT64_MAX = 0xFFFFFFFFFFFFFFFF


IF PTR_BITS == 64:
    cdef extern from "<stdint.h>":
        ctypedef char int8_t
        ctypedef unsigned char uint8_t

        ctypedef short int16_t
        ctypedef unsigned short uint16_t
        ctypedef int int32_t
        ctypedef unsigned int uint32_t
        ctypedef long long int64_t
        ctypedef unsigned long long uint64_t

        ctypedef long long intptr_t
        ctypedef unsigned long long uintptr_t
ELIF PTR_BITS == 32:
    cdef extern from "<stdint.h>":
        ctypedef char int8_t
        ctypedef unsigned char uint8_t

        ctypedef short int16_t
        ctypedef unsigned short uint16_t
        ctypedef int int32_t
        ctypedef unsigned int uint32_t
        ctypedef long long int64_t
        ctypedef unsigned long long uint64_t

        ctypedef int intptr_t
        ctypedef unsigned int uintptr_t
