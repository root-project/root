#ifndef ROOFITCOMPUTELIB_H
#define ROOFITCOMPUTELIB_H

#include "RooFitComputeInterface.h"

namespace RooFitCompute {

  namespace RF_ARCH {
    class RooFitComputeClass : RooFitComputeInterface {
      public:
        RooFitComputeClass();
        ~RooFitComputeClass() override {}
      };
  };
  
} // end namespace RooFitCompute

#endif
