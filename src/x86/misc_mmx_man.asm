

; /* hamming weight */

%ifndef REMOVE_DATA_SECTION
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
%endif
section .text

; manually compiled version
; size_t vec_u256n_get_hamming_weight(size_t size, uint8_t *src)
global vec_u256n_get_hamming_weight
vec_u256n_get_hamming_weight:
%ifdef USE_128BIT_UNITS
; {
%ifdef REMOVE_DATA_SECTION
    mov eax, 0x55555555
    mov edx, 0x33333333
    movd mm5, eax
    movd mm6, edx
    psllq mm5, 32
    psllq mm6, 32
    movd mm5, eax
    movd mm6, edx

    mov eax, 0x0F0F0F0F
    movd mm7, eax
    psllq mm7, 32
    movd mm7, eax
%endif

    ; size_t units = size / 16;
    mov edx, [esp + 4]

    ; __m64 *p = (__m64*)src;
    mov eax, [esp + 8]

    shr edx, 4

    ; __m64 rs = _mm_setzero_si64();
    pxor mm0, mm0

%ifndef REMOVE_DATA_SECTION
    movq mm5, [mask_u16x4_5555]
    movq mm6, [mask_u16x4_3333]
    movq mm7, [mask_u16x4_0F0F]
%endif

    ; for (size_t i = 0; i < units; ++i)
    or edx, edx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i * 2];
        ; __m64 it2 = p[i * 2 + 1];
        movq mm1, [eax]
        movq mm3, [eax + 8]
%macro vec_u256n_get_hamming_weight_i 0
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
%endmacro
        vec_u256n_get_hamming_weight_i
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
%ifdef REMOVE_DATA_SECTION
    mov eax, 0x55555555
    mov edx, 0x33333333
    movd mm3, eax
    movd mm4, edx
    psllq mm3, 32
    psllq mm4, 32
    movd mm3, eax
    movd mm4, edx

    mov eax, 0x0F0F0F0F
    mov edx, 0x00FF00FF
    movd mm5, eax
    movd mm6, edx
    psllq mm5, 32
    psllq mm6, 32
    movd mm5, eax
    movd mm6, edx

    mov eax, 0x0000FFFF
    movd mm7, eax
    psllq mm7, 32
    movd mm7, eax
%endif

    ; size_t units = size / 8;
    mov edx, [esp + 4]

    ; __m64 *p = (__m64*)src;
    mov eax, [esp + 8]

    shr edx, 3

    ; __m64 rs = _mm_setzero_si64();
    pxor mm0, mm0

%ifndef REMOVE_DATA_SECTION
    movq mm3, [mask_u16x4_5555]
    movq mm4, [mask_u16x4_3333]
    movq mm5, [mask_u16x4_0F0F]
    movq mm6, [mask_u16x4_00FF]
    movq mm7, [mask_u32x2_0000FFFF]
%endif

    ; for (size_t i = 0; i < units; ++i)
    or edx, edx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i];
        movq mm1, [eax]
%macro vec_u256n_get_hamming_weight_i 0
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
%endmacro
        vec_u256n_get_hamming_weight_i
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


; size_t vec_u256n_get_hamming_distance(size_t size, uint8_t *src1, uint8_t *src2)
global vec_u256n_get_hamming_distance
vec_u256n_get_hamming_distance:
%ifdef USE_128BIT_UNITS
; {
    push ecx

%ifdef REMOVE_DATA_SECTION
    mov eax, 0x55555555
    mov edx, 0x33333333
    mov ecx, 0x0F0F0F0F
    movd mm5, eax
    movd mm6, edx
    movd mm7, ecx
    psllq mm5, 32
    psllq mm6, 32
    psllq mm7, 32
    movd mm5, eax
    movd mm6, edx
    movd mm7, ecx
%endif

    ; size_t units = size / 16;
    mov ecx, [esp + 8]

    ; __m64 *p = (__m64*)src1;
    mov eax, [esp + 16]

    ; __m64 *q = (__m64*)src2;
    mov edx, [esp + 20]

    shr ecx, 4

    ; __m64 rs = _mm_setzero_si64();
    pxor mm0, mm0

%ifndef REMOVE_DATA_SECTION
    movq mm5, [mask_u16x4_5555]
    movq mm6, [mask_u16x4_3333]
    movq mm7, [mask_u16x4_0F0F]
%endif

    ; for (size_t i = 0; i < units; ++i)
    or ecx, ecx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i * 2];
        ; __m64 it2 = p[i * 2 + 1];
        ; __m64 it_b = q[i * 2];
        ; __m64 it2_b = q[i * 2 + 1];
        movq mm1, [eax]
        movq mm2, [edx]
        movq mm3, [eax + 8]
        movq mm4, [edx + 8]

        pxor mm1, mm2
        pxor mm3, mm4

        vec_u256n_get_hamming_weight_i
    ; }
        add eax, 16
        add edx, 16
        sub ecx, 1
        jnz .loop_start
.loop_end:

    movd eax, mm0
    psrlq mm0, 32
    movd edx, mm0

    add eax, edx

    emms

    ; return r;
    pop ecx
    ret
; }
%else
; {
    push ecx

%ifdef REMOVE_DATA_SECTION
    mov eax, 0x55555555
    mov edx, 0x33333333
    movd mm3, eax
    movd mm4, edx
    psllq mm3, 32
    psllq mm4, 32
    movd mm3, eax
    movd mm4, edx

    mov eax, 0x0F0F0F0F
    mov edx, 0x00FF00FF
    mov ecx, 0x0000FFFF
    movd mm5, eax
    movd mm6, edx
    movd mm7, ecx
    psllq mm5, 32
    psllq mm6, 32
    psllq mm7, 32
    movd mm5, eax
    movd mm6, edx
    movd mm7, ecx
%endif

    ; size_t units = size / 8;
    mov ecx, [esp + 8]

    ; __m64 *p = (__m64*)src1;
    mov eax, [esp + 16]

    ; __m64 *q = (__m64*)src2;
    mov edx, [esp + 20]

    shr ecx, 3

%ifndef REMOVE_DATA_SECTION
    movq mm3, [mask_u16x4_5555]
    movq mm4, [mask_u16x4_3333]
    movq mm5, [mask_u16x4_0F0F]
    movq mm6, [mask_u16x4_00FF]
    movq mm7, [mask_u32x2_0000FFFF]
%endif

    ; for (size_t i = 0; i < units; ++i)
    or ecx, ecx
    jz .loop_end
.loop_start:
    ;{
        ; __m64 it = p[i];
        ; __m64 it_b = q[i];
        movq mm1, [eax]
        movq mm2, [edx]

        pxor mm1, mm2

        vec_u256n_get_hamming_weight_i
    ; }
        add eax, 8
        add edx, 8
        sub ecx, 1
        jnz .loop_start
.loop_end:

    movd eax, mm0
    psrlq mm0, 32
    movd edx, mm0

    add eax, edx

    emms

    ; return r;
    pop ecx
    ret
; }
%endif
