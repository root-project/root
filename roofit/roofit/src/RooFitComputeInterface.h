#ifndef ROOFITCOMPUTEINTERFACE_H
#define ROOFITCOMPUTEINTERFACE_H

#include "RooSpan.h"
#include "DllImport.h" //for R__EXTERN, needed for windows

class RooAbsReal;
class RooListProxy;
namespace BatchHelpers {
  struct RunContext;
  class BracketAdapterWithMask;
}

namespace RooFitCompute {
  class RooFitComputeInterface {
  public:
    virtual ~RooFitComputeInterface() = default;
  };

  R__EXTERN RooFitComputeInterface * dispatch;
}

#endif
