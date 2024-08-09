// ispc --addressing=32 --arch=x86 -O3 --emit-obj -h minmax.h -o minmax.o minmax.ispc
// target: sse4.1-i16x8 sse4.2-i16x8 avx2-i16x16
#define target_size_t uint32
#define INT16_MAX 0x7FFF
#define INT16_MIN -32768


uniform int16 reduce_min(int16 src)
{
    uniform int16 result = INT16_MAX;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int16 it = extract(src, i);

        if (it < result) result = it;
    }

    return result;
}

uniform int16 reduce_max(int16 src)
{
    uniform int16 result = INT16_MIN;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int16 it = extract(src, i);

        if (it > result) result = it;
    }

    return result;
}

uniform int16 reduce_range(int16 src)
{
    return abs(reduce_max(src) - reduce_min(src));
}

uniform int16 reduce_range(uniform int16 src[], uniform target_size_t offsetBase)
{
    int16 it = src[offsetBase + programIndex];

    return reduce_max(it) - reduce_min(it);
}


#if TARGET_WIDTH > 16
#error "lager than 256bit SIMD is not supported"
#endif
export uniform int16 vec_i16x16n_get_min(uniform target_size_t size, uniform int16 src[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 mins = INT16_MAX;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int16 it = src[offsetBase + programIndex];

        mins = min(mins, it);
    }

    uniform int16 result = reduce_min(mins);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int16 it = src[vec_size + i];

        result = min(result, it);
    }

    return result;
}

export uniform int16 vec_i16x16n_get_max(uniform target_size_t size, uniform int16 src[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 maxs = INT16_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int16 it = src[offsetBase + programIndex];

        maxs = max(maxs, it);
    }

    uniform int16 result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int16 it = src[vec_size + i];

        result = max(result, it);
    }

    return result;
}

export void vec_i16x16n_get_minmax(uniform target_size_t size, uniform int16 src[], uniform int16 out_min[], uniform int16 out_max[])
{
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 mins = INT16_MAX;
    int16 maxs = INT16_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        int16 it = src[offsetBase + programIndex];

        mins = min(mins, it);
        maxs = max(maxs, it);
    }

    uniform int16 min_result = reduce_min(mins);
    uniform int16 max_result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform int16 it = src[vec_size + i];

        min_result = min(min_result, it);
        max_result = max(max_result, it);
    }

    *out_min = min_result;
    *out_max = max_result;
}


// get width of min ... max
export uniform int16 vec_i16x16n_get_range(uniform target_size_t size, uniform int16 src[])
{
    uniform int16 min_result;
    uniform int16 max_result;

    vec_i16x16n_get_minmax(size, src, &min_result, &max_result);

    return max_result - min_result;
}

// generates longer code
export uniform int16 vec_i16x16n_get_range_2(uniform target_size_t size, uniform int16 src[])
{
    return vec_i16x16n_get_max(size, src) - vec_i16x16n_get_min(size, src);
}

