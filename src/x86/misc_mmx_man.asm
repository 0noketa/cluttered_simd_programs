

; /* hamming weight */

section .data
align 8
mask_u16x4_5555: dd 0x55555555, 0x55555555
align 8
mask_u16x4_3333: dd 0x33333333, 0x33333333
align 8
mask_u16x4_0F0F: dd 0x0F0F0F0F, 0x0F0F0F0F
align 8
mask_u16x4_00FF: dd 0x00FF00FF, 0x00FF00FF
align 8
mask_u32x2_0000FFFF: dd 0x0000FFFF, 0x0000FFFF
section .text

; manually compiled version
; size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src)
global vec_u256n_get_hamming_weight
vec_u256n_get_hamming_weight:
%ifdef USE_128BIT_UNITS
; {
    ; size_t units = size / 16;
    mov edx, [esp + 4]
    shr edx, 4

    ; __m64 *p = (__m64*)src;
    mov eax, [esp + 8]

    ; __m64 rs = _mm_setzero_si64();
    pxor mm0, mm0

    movq mm5, [mask_u16x4_5555]
    movq mm6, [mask_u16x4_3333]
    movq mm7, [mask_u16x4_0F0F]

    ; for (size_t i = 0; i < units; ++i)
    or edx, edx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i * 2];
        ; __m64 it2 = p[i * 2 + 1];
        movq mm1, [eax]
        movq mm3, [eax + 8]

        ; __m64 tmp = _mm_srli_si64(it, 1);
        ; __m64 tmp2 = _mm_srli_si64(it2, 1);
        ; it = _mm_and_si64(it, mask);
        ; it2 = _mm_and_si64(it2, mask);
        ; tmp = _mm_and_si64(tmp, mask);
        ; tmp2 = _mm_and_si64(tmp2, mask);
        ; it = _mm_adds_pu16(it, tmp);
        ; it2 = _mm_adds_pu16(it2, tmp2);
        movq mm2, mm1
        movq mm4, mm3
        psrlq mm2, 1
        psrlq mm4, 1
        pand mm1, mm5
        pand mm2, mm5
        pand mm3, mm5
        pand mm4, mm5
        paddusw mm1, mm2
        paddusw mm3, mm4

        movq mm2, mm1
        movq mm4, mm3
        psrlq mm2, 2
        psrlq mm4, 2
        pand mm1, mm6
        pand mm2, mm6
        pand mm3, mm6
        pand mm4, mm6
        paddusw mm1, mm2
        paddusw mm3, mm4

        movq mm2, mm1
        movq mm4, mm3
        psrlq mm2, 4
        psrlq mm4, 4
        pand mm1, mm7
        pand mm2, mm7
        pand mm3, mm7
        pand mm4, mm7
        paddusw mm1, mm2
        paddusw mm3, mm4

        movq mm2, mm1
        movq mm4, mm3
        psrlw mm2, 8
        psrlw mm4, 8
        psllw mm1, 8
        psllw mm3, 8
        psrlw mm1, 8
        psrlw mm3, 8
        paddusw mm1, mm2
        paddusw mm3, mm4

        movq mm2, mm1
        movq mm4, mm3
        psrld mm2, 16
        psrld mm4, 16
        pslld mm1, 16
        pslld mm3, 16
        psrld mm1, 16
        psrld mm3, 16
        paddd mm1, mm2
        paddd mm3, mm4

        ; rs = _mm_add_pi32(rs, it);
        paddd mm0, mm1
        paddd mm0, mm3
    ; }
        add eax, 16
        sub edx, 1
        jnz .loop_start
.loop_end:

    movd eax, mm0
    psrlq mm0, 32
    movd edx, mm0

    add eax, edx

    emms

    ; return r;
    ret
; }
%else
; {
    ; size_t units = size / 8;
    mov edx, [esp + 4]
    shr edx, 3

    ; __m64 *p = (__m64*)src;
    mov eax, [esp + 8]

    ; __m64 rs = _mm_setzero_si64();
    pxor mm0, mm0

    movq mm3, [mask_u16x4_5555]
    movq mm4, [mask_u16x4_3333]
    movq mm5, [mask_u16x4_0F0F]
    movq mm6, [mask_u16x4_00FF]
    movq mm7, [mask_u32x2_0000FFFF]

    ; for (size_t i = 0; i < units; ++i)
    or edx, edx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i];
        movq mm1, [eax]

        ; __m64 tmp = _mm_srli_si64(it, 1);
        ; it = _mm_and_si64(it, mask);
        ; tmp = _mm_and_si64(tmp, mask);
        ; it = _mm_adds_pu16(it, tmp);
        movq mm2, mm1
        psrlq mm2, 1
        pand mm1, mm3
        pand mm2, mm3
        paddusw mm1, mm2

        movq mm2, mm1
        psrlq mm2, 2
        pand mm1, mm4
        pand mm2, mm4
        paddusw mm1, mm2

        movq mm2, mm1
        psrlq mm2, 4
        pand mm1, mm5
        pand mm2, mm5
        paddusw mm1, mm2

        movq mm2, mm1
        psrlq mm2, 8
        pand mm1, mm6
        pand mm2, mm6
        paddusw mm1, mm2

        movq mm2, mm1
        psrlq mm2, 16
        pand mm1, mm7
        pand mm2, mm7
        paddd mm1, mm2

        ; rs = _mm_add_pi32(rs, it);
        paddd mm0, mm1
    ; }
        add eax, 8
        sub edx, 1
        jnz .loop_start
.loop_end:

    movd eax, mm0
    psrlq mm0, 32
    movd edx, mm0

    add eax, edx

    emms

    ; return r;
    ret
; }
%endif

