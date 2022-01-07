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

// Note: components must be passed with std::move, otherwise it cannot be moved into the RooSumL because of the unique_ptr!
RooSumL::RooSumL(RooAbsPdf *pdf, RooAbsData *data, std::vector<std::unique_ptr<RooAbsL>> components, RooAbsL::Extended extended)
   : RooAbsL(pdf, data,
             data->numEntries(), // TODO: this may be misleading, because components in reality will have their own N_events...
             components.size(), extended), components_(std::move(components))
{}


bool RooSumL::processEmptyDataSets() const
{
   // TODO: check whether this is correct! This is copied the implementation of the RooNLLVar override; the
   // implementation in RooAbsTestStatistic always returns true
   return extended_;
}

double RooSumL::evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end)
{
   // Evaluate specified range of owned GOF objects
   double ret = 0;

   // from RooAbsOptTestStatistic::combinedValue (which is virtual, so could be different for non-RooNLLVar!):
   eval_carry_ = 0;
   for (std::size_t ix = components_begin; ix < components_end; ++ix) {
      // TODO: make sure we only calculate over events in the sub-range that the caller asked for
      //      std::size_t component_events_begin = std::max(events_begin, components_[ix]->get_N_events())  // THIS
      //      WON'T WORK, we need to somehow allow evaluate_partition to take in separate event ranges for all
      //      components...

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
