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
#include "Math/MinimizerOptions.h"

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;

class LikelihoodGradientWrapper {
public:
   explicit LikelihoodGradientWrapper(std::shared_ptr<RooAbsL> likelihood);
   virtual void fill_gradient(const double *x, double *grad) = 0;
   virtual void fill_second_derivative(const double *x, double *g2) = 0;
   virtual void fill_step_size(const double *x, double *gstep) = 0;

   // synchronize minimizer settings with calculators in child classes
   virtual void synchronize_with_minimizer(const ROOT::Math::MinimizerOptions & options);
private:
   std::shared_ptr<RooAbsL> likelihood;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodGradientWrapper
