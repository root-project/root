#ifndef ROOT_Math_VecTypes
#define ROOT_Math_VecTypes

#include "RConfigure.h"

#include "RtypesCore.h"

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD

#include <experimental/simd>

namespace ROOT {

namespace Internal {

#if defined(R__EXPERIMENTAL_SIMD_PIN_AVX_ABI) && defined(__AVX512F__)
// libstdc++'s <experimental/simd> _VecBltnBtmsk (AVX-512 mask) ABI fails to
// compile with non-GCC front ends (Clang, Intel icpx) due to a static_assert
// requiring `long long` and `long` to be the same type. When AVX-512 is
// enabled in this TU we'd otherwise hit that path, so pin to the 256-bit AVX
// ABI instead. The fallback is guarded by __AVX512F__ so that environments
// without AVX-512 in scope (notably rootcling/cling, which parses headers
// without -mavx*) still see the regular `native` ABI and pick the
// always-supported scalar fallback.
template <typename T>
using SIMDTag = std::experimental::simd_abi::__avx;
#else
template <typename T>
using SIMDTag = std::experimental::simd_abi::native<T>;
#endif

} // namespace Internal

// FIXME: Should we introduce Int32_t and UInt32_t in RtypesCore.h?
using Float_v = std::experimental::simd<Float_t, Internal::SIMDTag<Float_t>>;
using Double_v = std::experimental::simd<Double_t, Internal::SIMDTag<Double_t>>;
using Int_v = std::experimental::simd<Int_t, Internal::SIMDTag<Int_t>>;
using Int32_v = std::experimental::simd<Int_t, Internal::SIMDTag<Int_t>>;
using UInt_v = std::experimental::simd<UInt_t, Internal::SIMDTag<UInt_t>>;
using UInt32_v = std::experimental::simd<UInt_t, Internal::SIMDTag<UInt_t>>;

} // namespace ROOT

#endif

#endif // ROOT_Math_VecTypes
