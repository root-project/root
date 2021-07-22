// Author: Patrick Bos, Netherlands eScience Center / NIKHEF 2021

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2021, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include <TestStatistics/RooSumL.h>
#include <RooAbsData.h>
#include <TestStatistics/RooSubsidiaryL.h>

#include <algorithm> // min, max

namespace RooFit {
namespace TestStatistics {

/// \note Components must be passed with std::move, otherwise it cannot be moved into the RooSumL because of the unique_ptr!
/// \note The number of events in RooSumL is that of the full dataset. Components will have their own number of events that may be more relevant.
RooSumL::RooSumL(RooAbsPdf *pdf, RooAbsData *data, std::vector<std::unique_ptr<RooAbsL>> components, RooAbsL::Extended extended)
   : RooAbsL(pdf, data,
             data->numEntries(),
             components.size(), extended), components_(std::move(components))
{}

double RooSumL::evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end)
{
   // Evaluate specified range of owned GOF objects
   double ret = 0;

   // from RooAbsOptTestStatistic::combinedValue (which is virtual, so could be different for non-RooNLLVar!):
   eval_carry_ = 0;
   for (std::size_t ix = components_begin; ix < components_end; ++ix) {
      double y = components_[ix]->evaluatePartition(events, 0, 0);

      eval_carry_ += components_[ix]->getCarry();
      y -= eval_carry_;
      double t = ret + y;
      eval_carry_ = (t - ret) - y;
      ret = t;
   }

   // Note: compared to the RooAbsTestStatistic implementation that this was taken from, we leave out Hybrid and
   // SimComponents interleaving support here, this should be implemented by calculator, if desired.

   return ret;
}

// note: this assumes there is only one subsidiary component!
std::tuple<double, double> RooSumL::getSubsidiaryValue()
{
   // iterate in reverse, because the subsidiary component is usually at the end:
   for (auto component = components_.rbegin(); component != components_.rend(); ++component) {
      if (dynamic_cast<RooSubsidiaryL *>((*component).get()) != nullptr) {
         double value = (*component)->evaluatePartition({0, 1}, 0, 0);
         double carry = (*component)->getCarry();
         return {value, carry};
      }
   }
   return {0, 0};
}

void RooSumL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) {
   for (auto& component : components_) {
      component->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
   }
}

} // namespace TestStatistics
} // namespace RooFit
