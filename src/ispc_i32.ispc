// ispc --addressing=32 --arch=x86 -O3 --emit-obj -h minmax.h -o minmax.o minmax.ispc
// target: sse2-i32x8 sse4.1-i32x8 sse4.2-i32x8 avx2-i32x8
#define target_size_t uint32
#define INT32_MAX 0x7FFFFFFF
#define INT32_MIN -2147483648


#if TARGET_WIDTH > 8
#error "lager than 256bit SIMD is not supported"
#endif


uniform int32 reduce_range(int32 src)
{
    return reduce_max(src) - reduce_min(src);
}

uniform int32 reduce_range(uniform int32 src[], uniform target_size_t offsetBase)
{
    int32 it = src[offsetBase + programIndex];

    return reduce_max(it) - reduce_min(it);
}


export void vec_u32x8n_set_zero(uniform target_size_t size, uniform uint32 data[])
{
    for (uniform uint32 base = 0; base < size; base += TARGET_WIDTH)
    {
        dst[base + programIndex] = 0;
    }
}
export void vec_i32x8n_set_zero(uniform target_size_t size, uniform int32 data[])
{
    vec_u32v8n_set_zero(size, (int32*)data);
}
export void vec_u32x8n_set_iota(uniform target_size_t size, uniform uint32 dst[])
{
    for (uniform uint32 base = 0; base < size; base += TARGET_WIDTH)
    {
        dst[base + programIndex] = base + programIndex;
    }
}
// INT32_MAX + 1 as 0
export void vec_i32x8n_set_iota(uniform target_size_t size, uniform int32 dst[])
{
    for (uniform uint32 base = 0; base < size; base += TARGET_WIDTH)
    {
        uint32 it = base + programIndex;
        if (it > INT32_MAX) it %= INT32_MAX + 1;

        dst[base + programIndex] = it;
    }
}
// INT32_MAX + 1 as 0
export void vec_i32x8n_set_iota_saturated(uniform target_size_t size, uniform int32 dst[])
{
    for (uniform uint32 base = 0; base < size; base += TARGET_WIDTH)
    {
        uint32 it = base + programIndex;
        if (it > INT32_MAX) it = INT32_MAX;

        dst[base + programIndex] = it;
    }
}

// input: 8 columns decimal integer notation with zero padding
export uint32 nonvec_u32_get_parsed_int_col8(uniform uint8 src[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform uint32 result = 0;
    uniform uint32 base = i * TARGET_WIDTH;

#if TARGET_WIDTH >= 8
    if (programIndex < 8)
    {
        uint32 c = src[base + programIndex];

        c -= 48;

        if (programIndex == 0) c *= 1000000000;
        if (programIndex == 1) c *= 100000000;
        if (programIndex == 2) c *= 10000000;
        if (programIndex == 3) c *= 1000000;
        if (programIndex == 4) c *= 100000;
        if (programIndex == 5) c *= 10000;
        if (programIndex == 6) c *= 1000;
        if (programIndex == 7) c *= 100;

        result = reduce_add(c);
    }
#elif TARGET_WIDTH >= 4
    if (programIndex < 4)
    {
        uint32 c = src[base + programIndex];
        uint32 c2 = src[base + TARGET_WIDTH + programIndex];

        c -= 48;
        c2 -= 48;

        if (programIndex == 0) c *= 1000000000;
        if (programIndex == 1) c *= 100000000;
        if (programIndex == 2) c *= 10000000;
        if (programIndex == 3) c *= 1000000;
        if (programIndex == 0) c2 *= 100000;
        if (programIndex == 1) c2 *= 10000;
        if (programIndex == 2) c2 *= 1000;
        if (programIndex == 3) c2 *= 100;

        result = reduce_add(c) + reduce_add(c2);
    }
#elif TARGET_WIDTH == 2
    uint32 c = src[base + programIndex];
    uint32 c2 = src[base + TARGET_WIDTH + programIndex];
    uint32 c3 = src[base + TARGET_WIDTH * 2 + programIndex];
    uint32 c4 = src[base + TARGET_WIDTH * 3 + programIndex];

    c -= 48;
    c2 -= 48;
    c3 -= 48;
    c4 -= 48;

    if (programIndex == 0) c *= 1000000000;
    if (programIndex == 1) c *= 100000000;
    if (programIndex == 0) c2 *= 10000000;
    if (programIndex == 1) c2 *= 1000000;
    if (programIndex == 0) c3 *= 100000;
    if (programIndex == 1) c3 *= 10000;
    if (programIndex == 0) c4 *= 1000;
    if (programIndex == 1) c4 *= 100;

    result = reduce_add(c) + reduce_add(c2) + reduce_add(c3) + reduce_add(c4);
#else
#error "u32 SIMD is required"
#endif

    return result
}
// input: 10 columns decimal integer notation with zero padding
export uint32 nonvec_u32_get_parsed_int_col10(uniform uint8 src[])
{
    uniform uint32 base = i * TARGET_WIDTH;

    uniform uint32 c5 = src[base + TARGET_WIDTH * 2] - 48;
    uniform uint32 c6 = src[base + TARGET_WIDTH * 2 + 1] - 48;

    uniform uint32 result = nonvec_u32_get_parsed_int_col8(src);
    result += c5 + c6;

    return result
}



export void vec_u32x8n_replace_all(uniform target_size_t size, uniform uint32 dst[], uniform uint32 src[], uniform uint32 srcPtn, uniform uint32 dstPtn)
{
    for (uniform uint32 base = 0; base < length; base += TARGET_WIDTH)
    {
        uint32 v = src[base + programIndex];

        dst[base + programIndex] = v == srcPtn ? dstPtn : v;
    }
}
export void vec_u8x32n_replace_all(uniform target_size_t size, uniform uint32 dst[], uniform uint32 src[], uniform uint32 srcPtn, uniform uint32 dstPtn)
{
    for (uniform uint32 base = 0; base < length; base += TARGET_WIDTH)
    {
        uint32 vs = src[base + programIndex];

        dst[base + programIndex] =
            ((vs & 0x000000FF) == (srcPtn << 0) ? (dstPtn << 0) : (vs & 0x000000FF))
            | ((vs & 0x0000FF00) == (srcPtn << 8) ? (dstPtn << 8) : (vs & 0x0000FF00))
            | ((vs & 0x00FF0000) == (srcPtn << 16) ? (dstPtn << 16) : (vs & 0x00FF0000))
            | ((vs & 0xFF000000) == (srcPtn << 24) ? (dstPtn << 24) : (vs & 0xFF000000));
    }
}


export uniform int32 vec_i32v8n_min(uniform target_size_t size, uniform int32 src[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int32 mins = INT32_MAX;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int32 it = src[offsetBase + programIndex];

        mins = min(mins, it);
    }

    uniform int32 result = reduce_min(mins);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int32 it = src[vec_size + i];

        result = min(result, it);
    }

    return result;
}

export uniform int32 vec_i32v8n_max(uniform target_size_t size, uniform int32 src[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int32 maxs = INT32_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int32 it = src[offsetBase + programIndex];

        maxs = max(maxs, it);
    }

    uniform int32 result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int32 it = src[vec_size + i];

        result = max(result, it);
    }

    return result;
}

export void vec_i32v8n_get_minmax(uniform target_size_t size, uniform int32 src[], uniform int32 out_min[], uniform int32 out_max[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int32 mins = INT32_MAX;
    int32 maxs = INT32_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int32 it = src[offsetBase + programIndex];

        mins = min(mins, it);
        maxs = max(maxs, it);
    }

    uniform int32 min_result = reduce_min(mins);
    uniform int32 max_result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int32 it = src[vec_size + i];

        min_result = min(min_result, it);
        max_result = max(max_result, it);
    }

    *out_min = min_result;
    *out_max = max_result;
}


// get width of min ... max
export uniform int32 vec_i32v8n_get_range(uniform target_size_t size, uniform int32 src[])
{
    uniform int32 min_result;
    uniform int32 max_result;

    vec_i32v8n_get_minmax(size, src, &min_result, &max_result);

    return max_result - min_result;
}

// generates longer code
export uniform int32 vec_i32v8n_get_range_2(uniform target_size_t size, uniform int32 src[])
{
    return vec_i32v8n_max(size, src) - vec_i32v8n_min(size, src);
}
