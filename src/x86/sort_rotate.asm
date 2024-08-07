
section .text

global vec_i32v8n_inplace_reverse
global vec_i32v8n_reverse
global vec_i16v16n_inplace_reverse
global vec_i16v16n_reverse
global vec_i8v32n_inplace_reverse
global vec_i8v32n_reverse


vec_i32v8n_inplace_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; left
    mov esi, [esp + 3*4 + 2*4]

    ; right
    mov edi, esi
    shl ecx, 2
    add edi, ecx
    shr ecx, 2
    sub edi, 4

    shr ecx, 1

    .loop_top:
        mov eax, [esi]
        mov edx, [edi]
        mov [esi], edx
        mov [edi], eax

        add esi, 4
        sub edi, 4
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret

vec_i32v8n_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov edi, [esp + 3*4 + 3*4]
    shl ecx, 2
    add edi, ecx
    shr ecx, 2
    sub edi, 4

    shr ecx, 1

    .loop_top:
        mov eax, [esi + 0]
        mov edx, [esi + 4]
        mov [edi + 0], edx
        mov [edi + 4], eax

        add esi, 4
        sub edi, 4
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret

vec_i16v16n_inplace_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; left
    mov esi, [esp + 3*4 + 2*4]

    ; right
    mov edi, esi
    shl ecx, 1
    add edi, ecx
    shr ecx, 1
    sub edi, 4

    shr ecx, 2

    .loop_top:
        mov eax, [esi]
        mov edx, [edi]
        rol eax, 16
        rol edx, 16
        mov [esi], edx
        mov [edi], eax

        add esi, 4
        sub edi, 4
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret

vec_i16v16n_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov edi, [esp + 3*4 + 3*4]
    shl ecx, 1
    add edi, ecx
    shr ecx, 1
    sub edi, 8

    shr ecx, 2

    .loop_top:
        mov eax, [esi + 0]
        mov edx, [esi + 4]
        rol eax, 16
        rol edx, 16
        mov [edi + 0], edx
        mov [edi + 4], eax

        add esi, 8
        sub edi, 8
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret


vec_i8v32n_inplace_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; left
    mov esi, [esp + 3*4 + 2*4]

    ; right
    mov edi, esi
    add edi, ecx
    sub edi, 4

    shr ecx, 3

    .loop_top:
        mov eax, [esi]
        mov edx, [edi]
        rol ax, 8
        rol dx, 8
        rol eax, 16
        rol edx, 16
        rol ax, 8
        rol dx, 8
        mov [esi], edx
        mov [edi], eax

        add esi, 4
        sub edi, 4
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret

vec_i8v32n_reverse:
    push ecx
    push esi
    push edi

    ; size
    mov ecx, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov edi, [esp + 3*4 + 3*4]
    add edi, ecx
    sub edi, 8

    shr ecx, 3

    .loop_top:
        mov eax, [esi + 0]
        mov edx, [esi + 4]
        rol ax, 8
        rol dx, 8
        rol eax, 16
        rol edx, 16
        rol ax, 8
        rol dx, 8
        mov [edi + 0], edx
        mov [edi + 4], eax

        add esi, 8
        sub edi, 8
        sub ecx, 1
        jnz .loop_top

    pop edi
    pop esi
    pop ecx
    ret
