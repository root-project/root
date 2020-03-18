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
#ifndef ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
#define ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper

#include <memory>  // shared_ptr

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;

class LikelihoodGradientWrapper {
public:
   virtual double get_value(const double *x, std::size_t index);
private:
   std::shared_ptr<RooAbsL> likelihood;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
