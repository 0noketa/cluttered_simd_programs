# from libc.stdlib cimport malloc, free
include "cstdint.pxi"
from cmalloc cimport _aligned_malloc, _aligned_free
from sort cimport *
from search cimport *


BLOCK_SIZE = 32
ALIGNMENT = 32

ctypedef fused int_t:
    int8_t
    int16_t
    int32_t
ctypedef fused uint_t:
    uint8_t
    uint16_t
    uint32_t


cdef class Int8ArrayBase:
    cdef size_t len_
    cdef int8_t *arr_

    def __cinit__(self, sequence=None, size=-1):
        pass

    def __len__(self):
        return 0
    def __length_hint__(self):
        return 0
    def __getitem__(self, idx):
        return 0
    def __setitem__(self, idx, value):
        pass

cdef class Int8ArrayIter:
    cdef size_t idx_
    cdef size_t size_
    cdef int8_t *arr_
    cdef Int8ArrayBase obj_

    def __cinit__(self, Int8ArrayBase obj):
        self.idx_ = 0
        self.obj_ = obj
        self.size_ = obj.len_
        self.arr_ = obj.arr_

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx_ < self.size_:
            i = self.idx_
            self.idx_ += 1

            return self.arr_[i]

        raise StopIteration()

cdef class Int8Array(Int8ArrayBase):
    def __cinit__(self, sequence=None, size=-1):
        if sequence is not None:
            if size < 0:
                self.init_empty(len(sequence))
            else:
                self.init_empty(size)

            for i in range(self.len_):
                self.arr_[i] = sequence[i]
        else:
            if size < 0:
                size = BLOCK_SIZE

            self.init_empty(size)
    cdef init_empty(self, size_t size):
        self.len_ = size
        if self.len_ % BLOCK_SIZE != 0:
            self.len_ = self.len_ - self.len_ % BLOCK_SIZE + BLOCK_SIZE
        self.arr_ = <int8_t*>_aligned_malloc(sizeof(int8_t) * self.len_, ALIGNMENT)

    def __dealloc__(self):
        if self.arr_ is not NULL:
            _aligned_free(self.arr_)

    def __len__(self):
        return self.len_
    def __length_hint__(self):
        return self.len_

    def __iter__(self):
        return Int8ArrayIter(self)

    cdef int8_t get_(self, size_t idx):
        return self.arr_[idx % self.len_] if self.arr_ is not NULL else 0
    def __getitem__(self, idx):
        return self.get_(idx)

    cdef set_(self, size_t idx, int8_t value):
        if self.arr_ is not NULL:
            self.arr_[idx % self.len_] = value
    def __setitem__(self, idx, value):
        self.set_(idx, value)


    def reversed(self):
        cdef Int8Array dst = Int8Array(size=self.len_)
        vec_i8v32n_reverse(self.len_, self.arr_, dst.arr_)

        return dst

    def min(self):
        cdef int8_t dst = vec_i8v32n_get_min(self.len_, self.arr_)

        return dst

    def max(self):
        cdef int8_t dst = vec_i8v32n_get_max(self.len_, self.arr_)

        return dst

    def minmax(self):
        cdef int8_t min
        cdef int8_t max
        vec_i8v32n_get_minmax(self.len_, self.arr_, &min, &max)

        return (min, max)

    def get_sorted_indices_i8(self, int8_t value):
        cdef int8_t start
        cdef int8_t end
        vec_i8v32n_get_sorted_index(self.len_, self.arr_, value, &start, &end)

        return range(start, end)
    