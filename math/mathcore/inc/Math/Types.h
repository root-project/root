#ifndef ROOT_Math_VecTypes
#define ROOT_Math_VecTypes

#include "RConfigure.h"

#ifdef R__HAS_VECCORE

// We always try to use std::simd as the VecCore backend.
//
// If std::simd is not available, for example of AlmaLinux 8 with GCC 8.5, ROOT
// will just refuse to build with veccore=ON and give clear warnings at
// configuration time. We're sorry for the users who are affected by this, but
// the user base of the VecCore features in ROOT is small, and there are not
// many users sill on AlmaLinux 8 at this point. So this is acceptable
// collateral damage for the benefit of not having to test and support
// different VecCore backend types, e.g. also the old Vc.

#define ROOT_VECTORIZED_TMATH

#define VECCORE_ENABLE_STD_SIMD
#include <VecCore/VecCore>
#undef VECCORE_ENABLE_STD_SIMD

namespace ROOT {

namespace Internal {

using ScalarBackend = vecCore::backend::Scalar;
#ifdef ROOT_VECTORIZED_TMATH
// We can't use SIMDNative here, because we need ABI compatible types between
// the context of the interpreter and compiled code, for usage in the TFormula.
using VectorBackend = vecCore::backend::SIMD<std::experimental::simd_abi::compatible<double>>;
#else
using VectorBackend = vecCore::backend::Scalar;
#endif

} // namespace Internal

   using Float_v  = typename Internal::VectorBackend::Float_v;
   using Double_v = typename Internal::VectorBackend::Double_v;
   using Int_v    = typename Internal::VectorBackend::Int_v;
   using Int32_v  = typename Internal::VectorBackend::Int32_v;
   using UInt_v   = typename Internal::VectorBackend::UInt_v;
   using UInt32_v = typename Internal::VectorBackend::UInt32_v;
}

#else // R__HAS_VECCORE

// We do not have explicit vectorisation support enabled. Fall back to regular ROOT types.

#include "RtypesCore.h"

namespace ROOT {
   using Float_v  = Float_t;
   using Double_v = Double_t;
   using Int_v    = Int_t;
   using Int32_v  = Int_t; // FIXME: Should we introduce Int32_t in RtypesCore.h?
   using UInt_v   = UInt_t;
   using UInt32_v = UInt_t; // FIXME: Should we introduce UInt32_t in RtypesCore.h?
}
#endif

#endif // ROOT_Math_VecTypes
