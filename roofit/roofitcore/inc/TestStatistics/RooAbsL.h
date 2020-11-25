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
   enum class Extended {
      Yes, No, Auto
   };

   RooAbsL() = default;
   RooAbsL(RooAbsPdf *pdf, RooAbsData *data, bool do_offset, double offset, double offset_carry, std::size_t N_events,
           std::size_t N_components, Extended extended = Extended::Auto);
   RooAbsL(const RooAbsL& other);
   ~RooAbsL();

   void init_clones(RooAbsPdf& inpdf, RooAbsData& indata);

   virtual double evaluate_partition(std::size_t events_begin, std::size_t events_end, std::size_t components_begin,
                                     std::size_t components_end) = 0;
   virtual double get_carry() const = 0;

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

   std::size_t get_N_events() const;
   std::size_t get_N_components() const;

protected:
   virtual void optimize_pdf();
   std::unique_ptr<RooAbsPdf> pdf;
   std::unique_ptr<RooAbsData> data;
   RooArgSet *_normSet;      // Pointer to set with observables used for normalization
   bool _do_offset = false;
   double _offset = 0;
   double _offset_carry = 0;

   std::size_t N_events = 1;
   std::size_t N_components = 1;

   bool extended_ = false;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooAbsL
