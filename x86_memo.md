# confusing parts

## naming

``` c
/* pfix
_m_    : MMX (assembly styled mnemonic with exceptions)
_mm_   : MMX, SSE, AVX (abstraction)
_mm256_: AVX2 (256bit instraction)
*/
/* sfix
_si64 : 64-bit (unsigned) integer
_si128: 128-bit (unsigned) integer
_si256: 256-bit (unsigned) integer
_pi8  : packed 8-bit integers on MMX register
_pu8  : packed unsigned 8-bit integers on MMX register
_epi8 : packed 8-bit integers on XMM register (YMM with _mm256_ prefix)
_epu8 : packed unsigned 8-bit integers on XMM register (YMM with _mm256_ prefix)
*/
```

## order

``` c
__m64 m = _mm_set_pi32(higher, lower);
_mm_cvtsi64_si32(m) == lower;
_mm_cvtsi64_si32(_mm_srli_si64(m, 32)) == higher;

alignas(8) uint32_t arr[2] = { lower, higher );
__m64 m2 = *(__m64*)(void*)&arr; 
_mm_cvtsi64_si32(m2) == lower;
_mm_cvtsi64_si32(_mm_srli_si64(m2, 32)) == higher;

__m128i x = _mm_set_epi32(highest, _, _, lowest);
_mm_cvtsi128_si32(x) == lowest;
_mm_cvtsi128_si32(_mm_srli_si128(x, 12)) == highest;
```

## shift

on MMX, srl/sll for "si" uses bit-shift.
on SSE and AVX, srl/sll for "si" uses byte-shift.

``` c
_mm_srli_si64(x, 1);  // 1-bit
_mm_srli_si128(x, 1);  // 1-byte
_mm256_srli_si256(x, 1);  // 1-byte
```

in srl/sll for "pi", every lane shares COUNT. they use just one COUNT stored on lowest lane.

``` c
_mm_srli_pi32(x, 1);
// equals to
_mm_srl_pi32(x, _mm_set_pi32(_, 1));
```
