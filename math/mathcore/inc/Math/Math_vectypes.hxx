#include "RConfigure.h"

#ifdef R__HAS_VECCORE
 
#if defined(R__HAS_VC) && !defined(VECCORE_ENABLE_VC)
#define VECCORE_ENABLE_VC
#endif
 
#include <VecCore/VecCore>

namespace ROOT {

namespace Internal {
   using ScalarBackend = vecCore::backend::Scalar;
#ifdef VECCORE_ENABLE_VC
   using VectorBackend = vecCore::backend::VcVector;
#elif defined(VECCORE_ENABLE_UMESIMD)
   using VectorBackend = vecCore::backend::UMESimd;
#else
   using VectorBackend = vecCore::backend::Scalar;
#endif
}
   using Float_v  = typename Internal::VectorBackend::Float_v;
   using Double_v = typename Internal::VectorBackend::Double_v;
   using Int_v    = typename Internal::VectorBackend::Int_v;
   using Int32_v  = typename Internal::VectorBackend::Int32_v;
   using UInt_v   = typename Internal::VectorBackend::UInt_v;
   using UInt32_v = typename Internal::VectorBackend::UInt32_v;
}

#endif
