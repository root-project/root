include(CheckCXXSourceRuns)

# Helper function for checking if compiler is supporting AVX2 intrinsics
function(root_check_avx2)
    CHECK_CXX_COMPILER_FLAG("-mavx" AVX2_FLAG)
    CHECK_CXX_SOURCE_COMPILES("#include <immintrin.h> \n int main () {__m256i xmm =  _mm256_set1_epi64x(0); xmm =  _mm256_add_epi64(xmm,xmm);};" AVX2_COMPILATION)
    CHECK_CXX_SOURCE_RUNS("#include <immintrin.h> \n int main () {__m256i xmm =  _mm256_set1_epi64x(0); xmm =  _mm256_add_epi64(xmm,xmm);};" AVX2_RUN)
    if(AVX2_FLAG AND AVX2_COMPILATION AND AVX2_RUN)
      set(AVX2_SUPPORT TRUE PARENT_SCOPE)
    endif()
endfunction()

# Helper function for checking if compiler is supporting SSE4.1 intrinsics
function(root_check_sse41)
   CHECK_CXX_COMPILER_FLAG("-msse4.1" SSE_FLAG)
   CHECK_CXX_SOURCE_COMPILES("#include <smmintrin.h> \n int main () {__m128 xmm=_mm_set_ps1(0.0); _mm_ceil_ps(xmm);};" SSE_COMPILATION)
   CHECK_CXX_SOURCE_RUNS("#include <smmintrin.h> \n int main () {__m128 xmm=_mm_set_ps1(0.0); _mm_ceil_ps(xmm);};" SSE_RUN)
   if(SSE_FLAG AND SSE_COMPILATION AND SSE_RUN)
      set(SSE_SUPPORT TRUE PARENT_SCOPE)
   endif()
endfunction()
