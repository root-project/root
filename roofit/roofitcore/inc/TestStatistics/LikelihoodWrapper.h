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
#include <Fit/ParameterSettings.h>
#include "Math/MinimizerOptions.h"
#include "RooArgSet.h"
#include "RooAbsArg.h"  // enum ConstOpCode

namespace RooFit {
namespace TestStatistics {

// forward declaration
class RooAbsL;

class LikelihoodWrapper {
public:
   explicit LikelihoodWrapper(std::shared_ptr<RooAbsL> likelihood);
   virtual ~LikelihoodWrapper() = default;
   virtual double get_value(const double *x) = 0 ;

   // synchronize minimizer settings with calculators in child classes
   virtual void synchronize_with_minimizer(const ROOT::Math::MinimizerOptions & options);
   virtual void synchronize_parameter_settings(const std::vector<ROOT::Fit::ParameterSettings> &parameter_settings);

   // necessary from MinuitFcnGrad to reach likelihood properties:
   RooArgSet* getParameters();
   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode);
private:
   std::shared_ptr<RooAbsL> likelihood;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_LikelihoodWrapper
