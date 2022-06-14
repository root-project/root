/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSumL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSumL

#include <RooFit/TestStatistics/RooAbsL.h>
#include <RooFit/TestStatistics/optional_parameter_types.h>

#include "Math/Util.h" // KahanSum

#include <tuple>

namespace RooFit {
namespace TestStatistics {

class RooSumL : public RooAbsL {
public:
   RooSumL(RooAbsPdf *pdf, RooAbsData *data, std::vector<std::unique_ptr<RooAbsL>> components,
           RooAbsL::Extended extended = RooAbsL::Extended::Auto);
   // Note: when above ctor is called without std::moving components, you get a really obscure error. Pass as
   // std::move(components)!

   ROOT::Math::KahanSum<double>
   evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end) override;

   // necessary only for legacy offsetting mode in LikelihoodWrapper; TODO: remove this if legacy mode is ever removed
   ROOT::Math::KahanSum<double> getSubsidiaryValue();

   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override;

   virtual std::string GetClassName() const  override { return "RooSumL"; };

   const std::vector<std::unique_ptr<RooAbsL>>& GetComponents() const  { return components_; };

private:
   std::vector<std::unique_ptr<RooAbsL>> components_;
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSumL
