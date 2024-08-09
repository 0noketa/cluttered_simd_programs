
section .text

global vec_i32x8n_inplace_reverse
global vec_i32x8n_reverse
global vec_i16x16n_inplace_reverse
global vec_i16x16n_reverse
global vec_i8x32n_inplace_reverse
global vec_i8x32n_reverse


vec_i32x8n_inplace_reverse:
    push esi
    push edi
    push ebp
    mov ebp, esp

    ; size
    mov edi, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov eax, esi
    shl edi, 2
    add eax, edi
    shr edi, 2
    mov esp, eax

    shr edi, 1

    .loop_top:
        mov eax, [esi]
        mov edx, [esp - 4]
        push eax
        mov [esi], edx
        add esi, 4
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    ret

vec_i32x8n_reverse:
    push esi
    push edi
    push ebp
    mov ebp, esp

    ; size
    mov edi, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov eax, [esp + 3*4 + 3*4]
    shl edi, 2
    add eax, edi
    shr edi, 2
    mov esp, eax

    .loop_top:
        mov eax, [esi]
        push eax
        add esi, 4
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    ret


vec_i16x16n_inplace_reverse:
    push esi
    push edi
    push ebp
    mov ebp, esp

    ; size
    mov edi, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov eax, esi
    shl edi, 1
    add eax, edi
    shr edi, 1
    mov esp, eax

    shr edi, 2

    .loop_top:
        mov eax, [esi]
        mov edx, [esp - 4]
        rol eax, 16
        rol edx, 16
        push eax
        mov [esi], edx
        add esi, 4
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    ret

vec_i16x16n_reverse:
    push esi
    push edi
    push ebp
    ; save esp
    mov ebp, esp

    ; size
    mov edi, [esp + 3*4 + 1*4]

    ; src
    mov esi, [esp + 3*4 + 2*4]

    ; dst
    mov eax, [esp + 3*4 + 3*4]
    shl edi, 1
    add eax, edi
    shr edi, 1
    mov esp, eax

    shr edi, 1

    .loop_top:
        mov eax, [esi]
        rol eax, 16
        push eax
        add esi, 4
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    ret


vec_i8x32n_inplace_reverse:
    push ecx
    push ebx
    push esi
    push edi
    push ebp
    mov ebp, esp

    ; size
    mov edi, [esp + 5*4 + 1*4]

    ; left
    mov esi, [esp + 5*4 + 2*4]

    ; right
    mov eax, esi
    add eax, edi
    mov esp, eax

    shr edi, 4  ;/16

    .loop_top:
        mov eax, [esi + 0]
        mov ecx, [esi + 4]
        mov edx, [esp - 8]
        mov ebx, [esp - 4]
        rol ax, 8
        rol cx, 8
        rol dx, 8
        rol bx, 8
        rol eax, 16
        rol ecx, 16
        rol edx, 16
        rol ebx, 16
        rol ax, 8
        rol cx, 8
        rol dx, 8
        rol bx, 8
        push eax
        push ecx
        mov [esi + 4], edx
        mov [esi + 0], ebx
        add esi, 8
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    pop ebx
    pop ecx
    ret

vec_i8x32n_reverse:
    push ecx
    push ebx
    push esi
    push edi
    push ebp
    mov ebp, esp

    ; size
    mov edi, [esp + 5*4 + 1*4]

    ; src
    mov esi, [esp + 5*4 + 2*4]

    ; dst
    mov eax, [esp + 5*4 + 3*4]
    add eax, edi
    mov esp, eax

    shr edi, 4  ;/16

    .loop_top:
        mov eax, [esi + 0]
        mov ecx, [esi + 4]
        mov edx, [esi + 8]
        mov ebx, [esi + 12]
        rol ax, 8
        rol cx, 8
        rol dx, 8
        rol bx, 8
        rol eax, 16
        rol ecx, 16
        rol edx, 16
        rol ebx, 16
        rol ax, 8
        rol cx, 8
        rol dx, 8
        rol bx, 8
        push eax
        push ecx
        push edx
        push ebx
        add esi, 16
        sub edi, 1
        jnz .loop_top

    mov esp, ebp
    pop ebp
    pop edi
    pop esi
    pop ebx
    pop ecx
    ret

