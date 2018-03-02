

// define number for used to Mixmax
#define ROOT_MM_N 17

#include "MixMaxEngineImpl.h"

// define the template instance we want to have in the librsary
//( need to be declared as extern template in the .h file)
namespace ROOT {
   namespace Math {
      template class MixMaxEngine<17,0>;
      template class MixMaxEngine<17,1>;
      template class MixMaxEngine<17,2>;
   }
}
