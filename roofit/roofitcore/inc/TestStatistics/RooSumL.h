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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSumL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSumL

#include "TestStatistics/RooAbsL.h"
#include "TestStatistics/optional_parameter_types.h"

#include <tuple>

namespace RooFit {
namespace TestStatistics {

class RooSumL : public RooAbsL {
public:
   RooSumL(RooAbsPdf* pdf, RooAbsData* data, std::vector<std::unique_ptr<RooAbsL>> components,
           RooAbsL::Extended extended = RooAbsL::Extended::Auto);
   // Note: when above ctor is called without std::moving components, you get a really obscure error. Pass as std::move(components)!

   double evaluatePartition(Section events, std::size_t components_begin,
                             std::size_t components_end) override;

   // necessary only for legacy offsetting mode in LikelihoodWrapper; TODO: remove this if legacy mode is ever removed
   std::tuple<double, double> getSubsidiaryValue();

   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override;

private:
   std::vector<std::unique_ptr<RooAbsL>> components_;
};

}
}

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSumL
