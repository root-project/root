
#include "Math/StdEngine.h"

namespace ROOT {
   namespace Math {
      template class StdEngine<std::mt19937_64>; 
      template class StdEngine<std::ranlux48>; 
   }
}
