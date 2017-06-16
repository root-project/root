#include "RConfigure.h"
#ifdef R__HAS_VECCORE
 
  #ifdef R__HAS_VC
    #define VECCORE_ENABLE_VC 
  #endif
 
  #include <VecCore/VecCore>

  namespace ROOT {
    using Double_v = typename vecCore::backend::VcVector::Double_v;
  }
#endif
