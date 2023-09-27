// ispc --addressing=32 --arch=x86 -O3 --emit-obj -h minmax.h -o minmax.o minmax.ispc
// target: sse4.1-i8x16 sse4.2-i8x16 avx2-i8x32
#define target_size_t uint32
#define INT8_MAX 0x7F
#define INT8_MIN -128


uniform int8 reduce_min(int8 src)
{
    uniform int8 result = INT8_MAX;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int8 it = extract(src, i);

        if (it < result) result = it;
    }

    return result;
}

uniform int8 reduce_max(int8 src)
{
    uniform int8 result = INT8_MIN;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int8 it = extract(src, i);

        if (it > result) result = it;
    }

    return result;
}
uniform int16 reduce_min(int16 src)
{
    uniform int16 result = INT8_MAX;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int16 it = extract(src, i);

        if (it < result) result = it;
    }

    return result;
}

uniform int16 reduce_max(int16 src)
{
    uniform int16 result = INT8_MIN;

    for (uniform target_size_t i = 0; i < programCount; ++i)
    {
        uniform int16 it = extract(src, i);

        if (it > result) result = it;
    }

    return result;
}

uniform int8 reduce_range(int8 src)
{
    return reduce_max(src) - reduce_min(src);
}

uniform int8 reduce_range(uniform int8 src[], uniform target_size_t offsetBase)
{
    int8 it = src[offsetBase + programIndex];

    return reduce_max(it) - reduce_min(it);
}


#if TARGET_WIDTH > 32
#error "lager than 256bit SIMD is not supported"
#endif
export uniform int8 vec_i8v32n_get_min(uniform target_size_t size, uniform uint16 src[])
{
    size /= 2;
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 mins = INT8_MAX;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        uint16 it0 = src[offsetBase + programIndex];
        int16 it = (int8)(it0 & 0xFF);
        int16 it2 = (int8)(it0 >> 8);

        mins = min(mins, it);
        mins = min(mins, it2);
    }

    uniform int16 result = reduce_min(mins);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform uint16 it0 = src[vec_size + i];
        uniform int16 it = (int8)(it0 & 0xFF);
        uniform int16 it2 = (int8)(it0 >> 8);

        result = min(result, it);
        result = min(result, it2);
    }

    return result;
}

export uniform int8 vec_i8v32n_get_max(uniform target_size_t size, uniform uint16 src[])
{
    size /= 2;
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 maxs = INT8_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        uint16 it0 = src[offsetBase + programIndex];
        int16 it = (int8)(it0 & 0xFF);
        int16 it2 = (int8)(it0 >> 8);

        maxs = max(maxs, it);
        maxs = max(maxs, it2);
    }

    uniform int16 result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform uint16 it0 = src[vec_size + i];
        uniform int16 it = (int8)(it0 & 0xFF);
        uniform int16 it2 = (int8)(it0 >> 8);

        result = max(result, it);
        result = max(result, it2);
    }

    return result;
}

export void vec_i8v32n_get_minmax(uniform target_size_t size, uniform uint16 src[], uniform int8 out_min[], uniform int8 out_max[])
{
    size /= 2;
    uniform target_size_t units = size / TARGET_WIDTH;
    uniform target_size_t vec_size = units * TARGET_WIDTH;
    int16 mins = INT8_MAX;
    int16 maxs = INT8_MIN;

    for (uniform target_size_t i = 0; i < units; ++i)
    {
        uniform target_size_t offsetBase = i * TARGET_WIDTH;
        uint16 it0 = src[offsetBase + programIndex];
        int16 it = (int8)(it0 & 0xFF);
        int16 it2 = (int8)(it0 >> 8);

        mins = min(mins, it);
        maxs = max(maxs, it);
        mins = min(mins, it2);
        maxs = max(maxs, it2);
    }

    uniform int16 min_result = reduce_min(mins);
    uniform int16 max_result = reduce_max(maxs);

    for (uniform target_size_t i = vec_size; i < size; ++i)
    {
        uniform uint16 it0 = src[vec_size + i];
        uniform int16 it = (int8)(it0 & 0xFF);
        uniform int16 it2 = (int8)(it0 >> 8);

        min_result = min(min_result, it);
        max_result = max(max_result, it);
        min_result = min(min_result, it2);
        max_result = max(max_result, it2);
    }

    *out_min = min_result;
    *out_max = max_result;
}


// get width of min ... max
export uniform int8 vec_i8v32n_get_range(uniform target_size_t size, uniform uint16 src[])
{
    uniform int8 min_result;
    uniform int8 max_result;

    vec_i8v32n_get_minmax(size, src, &min_result, &max_result);

    return max_result - min_result;
}

// generates longer code
export uniform int8 vec_i8v32n_get_range_2(uniform target_size_t size, uniform uint16 src[])
{
    return vec_i8v32n_get_max(size, src) - vec_i8v32n_get_min(size, src);
}

