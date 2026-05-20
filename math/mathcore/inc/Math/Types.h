#ifndef ROOT_Math_VecTypes
#define ROOT_Math_VecTypes

#include "RConfigure.h"

#include "RtypesCore.h"

#ifdef R__HAS_STD_EXPERIMENTAL_SIMD

#include <experimental/simd>

namespace ROOT {

namespace Internal {

template <typename T>
using SIMDTag = std::experimental::simd_abi::native<T>;

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
