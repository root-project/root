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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL

#include <RooFit/TestStatistics/RooAbsL.h>
#include "RooArgList.h"
#include "RooArgSet.h"

#include "Math/Util.h" // KahanSum

namespace RooFit {
namespace TestStatistics {

class RooSubsidiaryL : public RooAbsL {
public:
   RooSubsidiaryL(const std::string &parent_pdf_name, const RooArgSet &pdfs, const RooArgSet &parameter_set);

   ROOT::Math::KahanSum<double>
   evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end) override;
   inline RooArgSet *getParameters() override { return &parameter_set_; }
   inline std::string GetName() const override { return std::string("subsidiary_pdf_of_") + parent_pdf_name_; }

   inline std::string GetTitle() const override
   {
      return std::string("Subsidiary PDF set of simultaneous PDF ") + parent_pdf_name_;
   }

   inline std::size_t numDataEntries() const override
   {
      // function only used in LikelihoodJob::evaluate, but this class must always be evaluated over Section(0,1), so
      // not useful
      return 0;
   }

   void constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt) override;

private:
   std::string parent_pdf_name_;
   RooArgList subsidiary_pdfs_{"subsidiary_pdfs"}; // Set of subsidiary PDF or "constraint" terms
   RooArgSet parameter_set_{"parameter_set"};      // Set of parameters to which constraints apply
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL
