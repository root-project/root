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

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL
#define ROOT_ROOFIT_TESTSTATISTICS_RooSubsidiaryL

#include <TestStatistics/RooAbsL.h>
#include <RooArgList.h>
#include <RooArgSet.h>

namespace RooFit {
namespace TestStatistics {

/// Gathers all subsidiary PDF terms from the component PDFs of RooSumL likelihoods.
/// These are summed separately for increased numerical stability, since these terms are often
/// small and cause numerical variances in their original PDFs, whereas by summing as one
/// separate subsidiary collective term, it is numerically very stable.
/// Note that when a subsidiary PDF is part of multiple component PDFs, it will only be summed
/// once in this class! This doesn't change the derivative of the log likelihood (which is what
/// matters in fitting the likelihood), but does change the value of the (log-)likelihood itself.
class RooSubsidiaryL : public RooAbsL {
public:
   RooSubsidiaryL(const std::string &parent_pdf_name, const RooArgSet &pdfs, const RooArgSet &parameter_set);

   double evaluatePartition(Section events, std::size_t components_begin, std::size_t components_end) override;
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
