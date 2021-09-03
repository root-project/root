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

#include <TestStatistics/RooAbsL.h>
#include <TestStatistics/ConstantTermsOptimizer.h>
#include "RooAbsPdf.h"
#include "RooAbsData.h"

// for dynamic casts in init_clones:
#include "RooAbsRealLValue.h"
#include "RooRealVar.h"
#include "RooDataHist.h"

// other stuff in init_clones:
#include "RooErrorHandler.h"
#include "RooMsgService.h"

// concrete classes in getParameters (testing, remove later)
#include "RooRealSumPdf.h"
#include "RooProdPdf.h"

namespace RooFit {
namespace TestStatistics {

// static function
bool RooAbsL::isExtendedHelper(RooAbsPdf* pdf, Extended extended)
{
   switch (extended) {
   case RooAbsL::Extended::No: {
      return false;
   }
   case RooAbsL::Extended::Yes: {
      return true;
   }
   case RooAbsL::Extended::Auto: {
      return ((pdf->extendMode() == RooAbsPdf::CanBeExtended || pdf->extendMode() == RooAbsPdf::MustBeExtended));
   }
   default: {
      throw std::logic_error("RooAbsL::isExtendedHelper got an unknown extended value!");
   }
   }
}

/// After handling cloning (or not) of the pdf and dataset, the public constructors call this private constructor to handle common tasks.
RooAbsL::RooAbsL(std::shared_ptr<RooAbsPdf> pdf, std::shared_ptr<RooAbsData> data,
                 std::size_t N_events, std::size_t N_components, Extended extended)
   : pdf_(std::move(pdf)), data_(std::move(data)), N_events_(N_events), N_components_(N_components)
{
   extended_ = isExtendedHelper(pdf_.get(), extended);
   if (extended == Extended::Auto) {
      if (extended_) {
         oocoutI((TObject *)nullptr, Minimization)
            << "in RooAbsL ctor: p.d.f. provides expected number of events, including extended term in likelihood."
            << std::endl;
      }
   }
}

/// Constructor that clones the pdf/data and owns those cloned copies.
///
/// This constructor is used for classes that need a pdf/data clone (RooBinnedL and RooUnbinnedL).
///
/// \param in Struct containing raw pointers to the pdf and dataset that are to be cloned.
/// \param N_events The number of events in this likelihood's dataset.
/// \param N_components The number of components in the likelihood.
/// \param extended Set extended term calculation on, off or use Extended::Auto to determine automatically based on the pdf whether to activate or not.
RooAbsL::RooAbsL(RooAbsL::ClonePdfData in, std::size_t N_events, std::size_t N_components, Extended extended)
  : RooAbsL(std::shared_ptr<RooAbsPdf>(static_cast<RooAbsPdf *>(in.pdf->cloneTree())),
     std::shared_ptr<RooAbsData>(static_cast<RooAbsData *>(in.data->Clone())), N_events, N_components, extended)
{
   initClones(*in.pdf, *in.data);
}

/// Constructor that does not clone pdf/data and uses the shared_ptr aliasing constructor to make it non-owning.
///
/// This constructor is used for classes where a reference to the external pdf/dataset is good enough (RooSumL and RooSubsidiaryL).
///
/// \param inpdf Raw pointer to the pdf.
/// \param indata Raw pointer to the dataset.
/// \param N_events The number of events in this likelihood's dataset.
/// \param N_components The number of components in the likelihood.
/// \param extended Set extended term calculation on, off or use Extended::Auto to determine automatically based on the pdf whether to activate or not.
RooAbsL::RooAbsL(RooAbsPdf *inpdf, RooAbsData *indata, std::size_t N_events, std::size_t N_components,
                 Extended extended)
   : RooAbsL({std::shared_ptr<RooAbsPdf>(nullptr), inpdf}, {std::shared_ptr<RooAbsData>(nullptr), indata}, N_events, N_components, extended)
{}


RooAbsL::RooAbsL(const RooAbsL &other)
   : pdf_(other.pdf_), data_(other.data_), N_events_(other.N_events_), N_components_(other.N_components_), extended_(other.extended_), sim_count_(other.sim_count_), eval_carry_(other.eval_carry_)
{
   // it can never be one, since we just copied the shared_ptr; if it is, something really weird is going on; also they must be equal (usually either zero or two)
   assert((pdf_.use_count() != 1) && (data_.use_count() != 1) && (pdf_.use_count() == data_.use_count()));
   if ((pdf_.use_count() > 1) && (data_.use_count() > 1)) {
      pdf_.reset(static_cast<RooAbsPdf *>(other.pdf_->cloneTree()));
      data_.reset(static_cast<RooAbsData *>(other.data_->Clone()));
      initClones(*other.pdf_, *other.data_);
   }
}

void RooAbsL::initClones(RooAbsPdf &inpdf, RooAbsData &indata)
{
   // ******************************************************************
   // *** PART 1 *** Clone incoming pdf, attach to each other *
   // ******************************************************************

   // Attach FUNC to data set
   auto _funcObsSet = pdf_->getObservables(indata);

   if (pdf_->getAttribute("BinnedLikelihood")) {
      pdf_->setAttribute("BinnedLikelihoodActive");
   }

   // Reattach FUNC to original parameters
   std::unique_ptr<RooArgSet> origParams{inpdf.getParameters(indata)};
   pdf_->recursiveRedirectServers(*origParams);

   // Store normalization set
   normSet_.reset((RooArgSet *)indata.get()->snapshot(kFALSE));

   // Expand list of observables with any observables used in parameterized ranges
   for (const auto realDep : *_funcObsSet) {
      auto realDepRLV = dynamic_cast<RooAbsRealLValue *>(realDep);
      if (realDepRLV && realDepRLV->isDerived()) {
         RooArgSet tmp2;
         realDepRLV->leafNodeServerList(&tmp2, 0, kTRUE);
         _funcObsSet->add(tmp2, kTRUE);
      }
   }

   // ******************************************************************
   // *** PART 2 *** Clone and adjust incoming data, attach to PDF     *
   // ******************************************************************

   // Check if the fit ranges of the dependents in the data and in the FUNC are consistent
   const RooArgSet *dataDepSet = indata.get();
   for (const auto arg : *_funcObsSet) {

      // Check that both dataset and function argument are of type RooRealVar
      auto realReal = dynamic_cast<RooRealVar *>(arg);
      if (!realReal) {
         continue;
      }
      auto datReal = dynamic_cast<RooRealVar *>(dataDepSet->find(realReal->GetName()));
      if (!datReal) {
         continue;
      }

      // Check that range of observables in pdf is equal or contained in range of observables in data

      if (!realReal->getBinning().lowBoundFunc() && realReal->getMin() < (datReal->getMin() - 1e-6)) {
         oocoutE((TObject *)0, InputArguments) << "RooAbsL: ERROR minimum of FUNC observable " << arg->GetName() << "("
                                               << realReal->getMin() << ") is smaller than that of " << arg->GetName()
                                               << " in the dataset (" << datReal->getMin() << ")" << std::endl;
         RooErrorHandler::softAbort();
         return;
      }

      if (!realReal->getBinning().highBoundFunc() && realReal->getMax() > (datReal->getMax() + 1e-6)) {
         oocoutE((TObject *)0, InputArguments)
            << "RooAbsL: ERROR maximum of FUNC observable " << arg->GetName() << " is larger than that of "
            << arg->GetName() << " in the dataset" << std::endl;
         RooErrorHandler::softAbort();
         return;
      }
   }

   // ******************************************************************
   // *** PART 3 *** Make adjustments for fit ranges, if specified     *
   // ******************************************************************

   // TODO

   // If dataset is binned, activate caching of bins that are invalid because they're outside the
   // updated range definition (WVE need to add virtual interface here)
   RooDataHist *tmph = dynamic_cast<RooDataHist *>(data_.get());
   if (tmph) {
      tmph->cacheValidEntries();
   }

   // This is deferred from part 2 - but must happen after part 3 - otherwise invalid bins cannot be properly marked in
   // cacheValidEntries
   data_->attachBuffers(*_funcObsSet);

   // *********************************************************************
   // *** PART 4 *** Adjust normalization range for projected observables *
   // *********************************************************************

   // TODO

   // *********************************************************************
   // *** PART 4 *** Finalization and activation of optimization          *
   // *********************************************************************

   // optimization steps (copied from ROATS::optimizeCaching)

   pdf_->getVal(normSet_.get());
   // Set value caching mode for all nodes that depend on any of the observables to ADirty
   pdf_->optimizeCacheMode(*_funcObsSet);
   // Disable propagation of dirty state flags for observables
   data_->setDirtyProp(kFALSE);

   // Disable reading of observables that are not used
   data_->optimizeReadingWithCaching(*pdf_, RooArgSet(), RooArgSet());
}

RooArgSet *RooAbsL::getParameters()
{
   auto ding = pdf_->getParameters(*data_);
   return ding;
}

void RooAbsL::constOptimizeTestStatistic(RooAbsArg::ConstOpCode opcode, bool doAlsoTrackingOpt)
{
   // to be further implemented, this is just a first test implementation
   if (opcode == RooAbsArg::Activate) {
      ConstantTermsOptimizer::enableConstantTermsOptimization(pdf_.get(), normSet_.get(), data_.get(), doAlsoTrackingOpt);
   }
}

std::string RooAbsL::GetName() const
{
   std::string output("likelihood of pdf ");
   output.append(pdf_->GetName());
   return output;
}

std::string RooAbsL::GetTitle() const
{
   std::string output("likelihood of pdf ");
   output.append(pdf_->GetTitle());
   return output;
}

std::size_t RooAbsL::numDataEntries() const
{
   return static_cast<std::size_t>(data_->numEntries());
}

} // namespace TestStatistics
} // namespace RooFit
