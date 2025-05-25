#ifndef BVH_V2_PLATFORM_H
#define BVH_V2_PLATFORM_H

#if defined(__clang__)
#define BVH_CLANG_ENABLE_FP_CONTRACT \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"") \
    _Pragma("STDC FP_CONTRACT ON") \
    _Pragma("clang diagnostic pop")
#else
#define BVH_CLANG_ENABLE_FP_CONTRACT
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BVH_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define BVH_ALWAYS_INLINE __forceinline
#else
#define BVH_ALWAYS_INLINE inline
#endif

#endif
