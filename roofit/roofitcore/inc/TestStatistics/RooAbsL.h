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
#include <string>
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
   virtual double get_carry() = 0;

   // necessary from MinuitFcnGrad to reach likelihood properties:
   RooArgSet *getParameters();
   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt);

   virtual std::string GetName() const;
   virtual std::string GetTitle() const;

   // necessary in RooMinimizer (via LikelihoodWrapper)
   virtual double defaultErrorLevel() const;

   // necessary in LikelihoodJob
   std::size_t numDataEntries() const;

   bool is_offsetting() const;
   void enable_offsetting(bool flag);

private:
   virtual void optimize_pdf();
   RooAbsPdf *pdf;
   RooAbsData *data;
   bool _do_offset = false;
   double _offset = 0;
   double _offset_carry = 0;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
