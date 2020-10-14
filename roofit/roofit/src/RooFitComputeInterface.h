#ifndef ROOFITCOMPUTEINTERFACE_H
#define ROOFITCOMPUTEINTERFACE_H

#include "RooSpan.h"

class RooAbsReal;
class RooListProxy;
namespace BatchHelpers {
  struct RunContext;
  class BracketAdapterWithMask;
}

namespace RooFitCompute {
  class RooFitComputeInterface {
  public:
    virtual ~RooFitComputeInterface() {}
  };

  extern RooFitComputeInterface* dispatch;
}

#endif
