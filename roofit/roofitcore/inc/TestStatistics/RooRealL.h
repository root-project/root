/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper

#include <memory>  // shared_ptr
#include <RooAbsReal.h>
#include "RooListProxy.h"

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;

class RooRealL : RooAbsReal {
private:
   std::shared_ptr<RooAbsL> likelihood;

   // TODO: we need to track the clean/dirty state in this wrapper. See the old RooRealMPFE implementation for how that can be done automatically using the RooListProxy.
   RooListProxy _vars;    // Variables
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper
