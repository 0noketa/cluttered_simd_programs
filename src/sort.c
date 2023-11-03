#if defined(__AVX2__)
#   include "./x86/sort_avx2.c"
#elif defined(__SSE2__) //|| defined(__x86_64__) || defined(_WIN64)
#   include "./x86/sort_sse2.c"
#elif defined(__MMX__) //|| defined(__i386__) || defined(_WIN32)
#   include "./x86/sort_mmx.c"
#else
#   include "./generic/sort.c"
#endif
