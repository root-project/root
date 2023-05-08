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

/**
\file RooSubsidiaryL.cxx
\class RooSubsidiaryL
\ingroup Roofitcore

\brief RooSubsidiaryL calculates the sum of the -(log) likelihoods of a set of RooAbsPdf objects that represent
subsidiary or constraint functions.

This class is used to gather all subsidiary PDF terms from the component PDFs of RooSumL likelihoods and calculate the
composite -log(L). Such subsidiary terms can be marked using RooFit::Constrain arguments to RooAbsPdf::fitTo() or
RooAbsPdf::createNLL().

Subsidiary terms are summed separately for increased numerical stability, since these terms are often small and cause
numerical variances in their original PDFs, whereas by summing as one separate subsidiary collective term, it is
numerically very stable.

\note When a subsidiary PDF is part of multiple component PDFs, it will only be summed once in this class! This doesn't
change the derivative of the log likelihood (which is what matters in fitting the likelihood), but does change the
value of the (log-)likelihood itself.
**/

#include <RooFit/TestStatistics/RooSubsidiaryL.h>
#include <RooAbsPdf.h> // for dynamic cast
#include <RooErrorHandler.h>

#include "Math/Util.h" // KahanSum

namespace RooFit {
namespace TestStatistics {

RooSubsidiaryL::RooSubsidiaryL(const std::string &parent_pdf_name, const RooArgSet &pdfs,
                               const RooArgSet &parameter_set)
   : RooAbsL(nullptr, nullptr, 0, 0, RooAbsL::Extended::No), parent_pdf_name_(parent_pdf_name)
{
   for (const auto comp : pdfs) {
      if (!dynamic_cast<RooAbsPdf *>(comp)) {
         oocoutE(nullptr, InputArguments) << "RooSubsidiaryL::ctor(" << GetName() << ") ERROR: component "
                                               << comp->GetName() << " is not of type RooAbsPdf" << std::endl;
         RooErrorHandler::softAbort();
      }
      subsidiary_pdfs_.add(*comp);
   }
   parameter_set_.add(parameter_set);
}

/// \note The subsidiary term is only calculated together with the last event.
///       While this is meaningless for the subsidiary term itself (it has no
///       events), it is useful when calculating RooSumLs by parts. The Section
///       from each part is forwarded here if the component is a RooSubsidiaryL.
ROOT::Math::KahanSum<double> RooSubsidiaryL::evaluatePartition(RooAbsL::Section events,
                                                               std::size_t /*components_begin*/,
                                                               std::size_t /*components_end*/)
{
   ROOT::Math::KahanSum<double> sum;

   if (events.end_fraction == 1) {
      for (const auto comp : subsidiary_pdfs_) {
         sum += -static_cast<RooAbsPdf*>(comp)->getLogVal(&parameter_set_);
      }
   }

   return sum;
}

void RooSubsidiaryL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/) {}

} // namespace TestStatistics
} // namespace RooFit
