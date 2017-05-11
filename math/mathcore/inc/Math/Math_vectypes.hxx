#include "RConfigure.h"
#ifdef R__HAS_VECCORE
#include <VecCore/VecCore>
using Double_v = typename vecCore::backend::VcVector::Double_v;
#endif
