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
#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
#define ROOT_ROOFIT_TESTSTATISTICS_RooAbsL

#include <cstddef> // std::size_t
#include "RooArgSet.h"
#include "RooAbsArg.h" // enum ConstOpCode

// forward declarations
class RooAbsPdf;
class RooAbsData;

namespace RooFit {
namespace TestStatistics {

class RooAbsL {
public:
   virtual double evaluate_partition(std::size_t, std::size_t, std::size_t) = 0;

   // necessary from MinuitFcnGrad to reach likelihood properties:
   RooArgSet *getParameters();
   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode);

private:
   virtual void optimize_pdf();
   RooAbsPdf *pdf;
   RooAbsData *data;
   //   RooPDFOptimizer optimizer;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
