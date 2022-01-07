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

#include <TestStatistics/likelihood_builders.h>

#include <RooSimultaneous.h>
#include <TestStatistics/RooBinnedL.h>
#include <TestStatistics/RooUnbinnedL.h>
#include <TestStatistics/RooSubsidiaryL.h>
#include <TestStatistics/RooSumL.h>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooRealSumPdf.h>
#include <RooProdPdf.h>

#include <memory>


namespace RooFit {
/**
 * \brief Namespace for new RooFit test statistic calculation.
 *
 * RooFit::TestStatistics contains a major refactoring of the RooAbsTestStatistic-RooAbsOptTestStatistic-RooNLLVar inheritance tree into:
 *   1. statistics-based classes on the one hand;
 *   2. calculation/evaluation/optimization based classes on the other hand.
 *
 * The likelihood is the central unit on the statistics side. The RooAbsL class is implemented for four kinds of likelihoods:
 * binned, unbinned, "subsidiary" (an optimization for numerical stability that gathers components like global observables)
 * and "sum" (over multiple components of the other types). These classes provide ways to compute their components in parallelizable
 * chunks that can be used by the calculator classes as they see fit.
 *
 * On top of the likelihood classes, we also provide for convenience a set of likelihood builders, as free functions in the namespace.
 *
 * The calculator "Wrapper" classes are abstract interfaces. These can be implemented for different kinds of algorithms, or with
 * different kinds of optimization "back-ends" in mind. In an upcoming PR, we will introduce the fork-based multi-processing
 * implementation based on RooFit::MultiProcess. Other possible implementations could use the GPU or external tools like TensorFlow.
 *
 * The coupling of all these classes to RooMinimizer is made via the MinuitFcnGrad class, which owns the Wrappers that calculate
 * the likelihood components.
 */
namespace TestStatistics {

/*
 * \brief Extract a collection of subsidiary likelihoods from a pdf
 *
 * \param[in] pdf Raw pointer to the pdf
 * \param[in] data Raw pointer to the dataset
 * \param[in] constrained_parameters Set of parameters that are constrained. Pdf components dependent on these alone are added to the subsidiary likelihood.
 * \param[in] external_constraints Set of external constraint pdfs, i.e. constraints not necessarily in the pdf itself. These are always added to the subsidiary likelihood.
 * \param[in] global_observables Observables that have a constant value, independent of the dataset events. Pdf components dependent on these alone are added to the subsidiary likelihood. \note Overrides all other likelihood parameters (like those in \p constrained_parameters) if present.
 * \param[in] global_observables_tag String that can be set as attribute in pdf components to indicate that it is a global observable. Can be used instead of or in addition to \p global_observables.
 * \return A unique pointer to a RooSubsidiaryL that contains all terms in the pdf that can be calculated separately from the other components in the full likelihood.
 */
std::unique_ptr<RooSubsidiaryL> buildConstraints(RooAbsPdf *pdf, RooAbsData *data,
                                                 ConstrainedParameters constrained_parameters, ExternalConstraints external_constraints,
                                                 GlobalObservables global_observables, std::string global_observables_tag)
{
   // BEGIN CONSTRAINT COLLECTION; copied from RooAbsPdf::createNLL

   Bool_t doStripDisconnected = kFALSE;
   // If no explicit list of parameters to be constrained is specified apply default algorithm
   // All terms of RooProdPdfs that do not contain observables and share parameters with one or more
   // terms that do contain observables are added as constraints.
#ifndef NDEBUG
   bool did_default_constraint_algo = false;
   std::size_t N_default_constraints = 0;
#endif
   if (constrained_parameters.set.getSize() == 0) {
      std::unique_ptr<RooArgSet> default_constraints{pdf->getParameters(*data, kFALSE)};
      constrained_parameters.set.add(*default_constraints);
      doStripDisconnected = kTRUE;
#ifndef NDEBUG
      did_default_constraint_algo = true;
      N_default_constraints = default_constraints->getSize();
#endif
   }
#ifndef NDEBUG
   if (did_default_constraint_algo) {
      assert(N_default_constraints == static_cast<std::size_t>(constrained_parameters.set.getSize()));
   }
#endif

   // Collect internal and external constraint specifications
   RooArgSet allConstraints;

   if (!global_observables_tag.empty()) {
      if (global_observables.set.getSize() > 0) {
         global_observables.set.removeAll();
      }
      std::unique_ptr<RooArgSet> allVars {pdf->getVariables()};
      global_observables.set.add(*dynamic_cast<RooArgSet *>(allVars->selectByAttrib(global_observables_tag.c_str(), kTRUE)));
      oocoutI((TObject*)nullptr, Minimization) << "User-defined specification of global observables definition with tag named '" <<  global_observables_tag << "'" << std::endl;
   } else if (global_observables.set.getSize() == 0) {
      // neither global_observables nor global_observables_tag was given - try if a default tag is defined in the head node
      const char* defGlobObsTag = pdf->getStringAttribute("DefaultGlobalObservablesTag");
      if (defGlobObsTag) {
         oocoutI((TObject*)nullptr, Minimization) << "p.d.f. provides built-in specification of global observables definition with tag named '" <<  defGlobObsTag << "'" << std::endl;
         std::unique_ptr<RooArgSet> allVars {pdf->getVariables()};
         global_observables.set.add(*dynamic_cast<RooArgSet *>(allVars->selectByAttrib(defGlobObsTag, kTRUE)));
      }
   }

   // EGP: removed workspace (RooAbsPdf::_myws) based stuff for now; TODO: reconnect this class to workspaces

   if (constrained_parameters.set.getSize() > 0) {
      std::unique_ptr<RooArgSet> constraints{pdf->getAllConstraints(*data->get(), constrained_parameters.set, doStripDisconnected)};
      allConstraints.add(*constraints);
   }
   if (external_constraints.set.getSize() > 0) {
      allConstraints.add(external_constraints.set);
   }

   std::unique_ptr<RooSubsidiaryL> subsidiary_likelihood;
   // Include constraints, if any, in likelihood
   if (allConstraints.getSize() > 0) {

      oocoutI((TObject*) nullptr, Minimization) << " Including the following contraint terms in minimization: " << allConstraints << std::endl;
      if (global_observables.set.getSize() > 0) {
         oocoutI((TObject*) nullptr, Minimization) << "The following global observables have been defined: " << global_observables.set << std::endl;
      }
      std::string name("likelihood for pdf ");
      name += pdf->GetName();
      subsidiary_likelihood = std::make_unique<RooSubsidiaryL>(name, allConstraints,
                                                               (global_observables.set.getSize() > 0) ? global_observables.set : constrained_parameters.set);
   }

   // END CONSTRAINT COLLECTION; copied from RooAbsPdf::createNLL

   return subsidiary_likelihood;
}


/*
 * \brief Build a likelihood from a simultaneous pdf, possibly including subsidiary likelihood component
 *
 * \param[in] pdf Raw pointer to the pdf
 * \param[in] data Raw pointer to the dataset
 * \param[in] extended Set extended term calculation on, off or use Extended::Auto to determine automatically based on the pdf whether to activate or not.
 * \param[in] constrained_parameters Set of parameters that are constrained. Pdf components dependent on these alone are added to the subsidiary likelihood.
 * \param[in] external_constraints Set of external constraint pdfs, i.e. constraints not necessarily in the pdf itself. These are always added to the subsidiary likelihood.
 * \param[in] global_observables Observables that have a constant value, independent of the dataset events. Pdf components dependent on these alone are added to the subsidiary likelihood. \note Overrides all other likelihood parameters (like those in \p constrained_parameters) if present.
 * \param[in] global_observables_tag String that can be set as attribute in pdf components to indicate that it is a global observable. Can be used instead of or in addition to \p global_observables.
 * \return A unique pointer to a RooSubsidiaryL that contains all terms in the pdf that can be calculated separately from the other components in the full likelihood.
 */
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended,
                                                       ConstrainedParameters constrained_parameters, ExternalConstraints external_constraints,
                                                       GlobalObservables global_observables, std::string global_observables_tag) {
   auto sim_pdf = dynamic_cast<RooSimultaneous *>(pdf);
   if (sim_pdf == nullptr) {
      throw std::logic_error("Can only build RooSumL from RooSimultaneous pdf!");
   }

   // the rest of this function is an adaptation of RooAbsTestStatistic::initSimMode:

   RooAbsCategoryLValue &simCat = (RooAbsCategoryLValue &)sim_pdf->indexCat();

   // note: this is valid for simultaneous likelihoods, not for other test statistic types (e.g. chi2) for which this should return true.
   bool process_empty_data_sets = RooAbsL::isExtendedHelper(pdf, extended);

   TString simCatName(simCat.GetName());
   // Note: important not to use cloned dataset here (possible when this code is run in Roo[...]L ctor), use the
   // original one (which is data_ in Roo[...]L ctors, but data here)
   std::unique_ptr<TList> dsetList{data->split(simCat, process_empty_data_sets)};
   if (!dsetList) {
      throw std::logic_error("buildSimultaneousLikelihood ERROR, index category of simultaneous pdf is missing in dataset, aborting");
   }

   // Count number of used states
   std::size_t N_components = 0;

   for (const auto& catState : simCat) {
      // Retrieve the PDF for this simCat state
      RooAbsPdf *component_pdf = sim_pdf->getPdf(catState.first.c_str());
      auto dset = (RooAbsData *)dsetList->FindObject(catState.first.c_str());

      if (component_pdf && dset && (0. != dset->sumEntries() || process_empty_data_sets)) {
         ++N_components;
      }
   }

   // Allocate arrays
   std::vector<std::unique_ptr<RooAbsL>> components;
   components.reserve(N_components);
   //   _gofSplitMode.resize(N_components);  // not used, Hybrid mode only, see below

   // Create array of regular fit contexts, containing subset of data and single fitCat PDF
   std::size_t n = 0;
   for (const auto& catState : simCat) {
      const std::string& catName = catState.first;
      // Retrieve the PDF for this simCat state
      RooAbsPdf *component_pdf = sim_pdf->getPdf(catName.c_str());
      auto dset = (RooAbsData *)dsetList->FindObject(catName.c_str());

      if (component_pdf && dset && (0. != dset->sumEntries() || process_empty_data_sets)) {
         ooccoutI((TObject *)nullptr, Fitting)
         << "RooSumL: creating slave calculator #" << n << " for state " << catName << " ("
                                                   << dset->numEntries() << " dataset entries)" << std::endl;

         // *** START HERE
         // WVE HACK determine if we have a RooRealSumPdf and then treat it like a binned likelihood
         RooAbsPdf *binnedPdf = 0;
         Bool_t binnedL = kFALSE;
         if (component_pdf->getAttribute("BinnedLikelihood") &&
             component_pdf->IsA()->InheritsFrom(RooRealSumPdf::Class())) {
            // Simplest case: top-level of component is a RRSP
            binnedPdf = component_pdf;
            binnedL = kTRUE;
         } else if (component_pdf->IsA()->InheritsFrom(RooProdPdf::Class())) {
            // Default case: top-level pdf is a product of RRSP and other pdfs
            for (const auto component : ((RooProdPdf *)component_pdf)->pdfList()) {
               if (component->getAttribute("BinnedLikelihood") &&
                   component->IsA()->InheritsFrom(RooRealSumPdf::Class())) {
                  binnedPdf = (RooAbsPdf *)component;
                  binnedL = kTRUE;
               }
               if (component->getAttribute("MAIN_MEASUREMENT")) {
                  // not really a binned pdf, but this prevents a (potentially) long list of subsidiary measurements to
                  // be passed to the slave calculator
                  binnedPdf = (RooAbsPdf *)component;
               }
            }
         }
         // WVE END HACK
         // Below here directly pass binnedPdf instead of PROD(binnedPdf,constraints) as constraints are evaluated
         // elsewhere anyway and omitting them reduces model complexity and associated handling/cloning times
         if (binnedL) {
            components.push_back(std::make_unique<RooBinnedL>((binnedPdf ? binnedPdf : component_pdf), dset));
         } else {
            components.push_back(std::make_unique<RooUnbinnedL>((binnedPdf ? binnedPdf : component_pdf), dset));
         }
         //         }
         components.back()->setSimCount(N_components);
         // *** END HERE

         // Servers may have been redirected between instantiation and (deferred) initialization

         std::unique_ptr<RooArgSet> actualParams{binnedPdf ? binnedPdf->getParameters(dset)
                                                           : component_pdf->getParameters(dset)};
         std::unique_ptr<RooArgSet> selTargetParams{(RooArgSet *)pdf->getParameters(*data)->selectCommon(*actualParams)};

         // TODO: I don't think we have to redirect servers, because our classes make no use of those, but we should
         // make sure. Do we need to reset the parameter set instead?
         //         components_.back()->recursiveRedirectServers(*selTargetParams);
         assert(selTargetParams->equals(*components.back()->getParameters()));

         ++n;
      } else {
         if ((!dset || (0. != dset->sumEntries() && !process_empty_data_sets)) && component_pdf) {
            ooccoutD((TObject *)nullptr, Fitting) << "RooSumL: state " << catName
                                                                       << " has no data entries, no slave calculator created" << std::endl;
         }
      }
   }
   oocoutI((TObject *)nullptr, Fitting) << "RooSumL: created " << n << " slave calculators." << std::endl;

   std::unique_ptr<RooAbsL> subsidiary = buildConstraints(pdf, data, constrained_parameters, external_constraints, global_observables, global_observables_tag);
   if (subsidiary) {
      components.push_back(std::move(subsidiary));
   }

   return std::make_shared<RooSumL>(pdf, data, std::move(components), extended);
}

// delegating convenience overloads
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters)
{
   return buildSimultaneousLikelihood(pdf, data, RooAbsL::Extended::Auto, constrained_parameters);
}
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints)
{
   return buildSimultaneousLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, external_constraints);
}
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables)
{
   return buildSimultaneousLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, {}, global_observables);
}
std::shared_ptr<RooAbsL>
buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag)
{
   return buildSimultaneousLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, {}, {}, global_observables_tag);
}
std::shared_ptr<RooAbsL> buildSimultaneousLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters, GlobalObservables global_observables)
{
   return buildSimultaneousLikelihood(pdf, data, RooAbsL::Extended::Auto, constrained_parameters, {},
                                      global_observables);
}


/*
 * \brief Build a likelihood from an unbinned pdf with a subsidiary likelihood component
 *
 * \param[in] pdf Raw pointer to the pdf
 * \param[in] data Raw pointer to the dataset
 * \param[in] extended Set extended term calculation on, off or use Extended::Auto to determine automatically based on the pdf whether to activate or not.
 * \param[in] constrained_parameters Set of parameters that are constrained. Pdf components dependent on these alone are added to the subsidiary likelihood.
 * \param[in] external_constraints Set of external constraint pdfs, i.e. constraints not necessarily in the pdf itself. These are always added to the subsidiary likelihood.
 * \param[in] global_observables Observables that have a constant value, independent of the dataset events. Pdf components dependent on these alone are added to the subsidiary likelihood. \note Overrides all other likelihood parameters (like those in \p constrained_parameters) if present.
 * \param[in] global_observables_tag String that can be set as attribute in pdf components to indicate that it is a global observable. Can be used instead of or in addition to \p global_observables.
 * \return A unique pointer to a RooSubsidiaryL that contains all terms in the pdf that can be calculated separately from the other components in the full likelihood.
 */
std::shared_ptr<RooAbsL>
buildUnbinnedConstrainedLikelihood(RooAbsPdf *pdf, RooAbsData *data, RooAbsL::Extended extended,
                                                               ConstrainedParameters constrained_parameters, ExternalConstraints external_constraints,
                                                               GlobalObservables global_observables, std::string global_observables_tag) {
   std::vector<std::unique_ptr<RooAbsL>> components;
   components.reserve(2);
   components.push_back(std::make_unique<RooUnbinnedL>(pdf, data, extended));
   components.push_back(buildConstraints(pdf, data, constrained_parameters, external_constraints, global_observables, global_observables_tag));
   return std::make_shared<RooSumL>(pdf, data, std::move(components), extended);
}

// delegating convenience overloads
std::shared_ptr<RooAbsL>
buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, ConstrainedParameters constrained_parameters)
{
   return buildUnbinnedConstrainedLikelihood(pdf, data, RooAbsL::Extended::Auto, constrained_parameters);
}
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, ExternalConstraints external_constraints)
{
   return buildUnbinnedConstrainedLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, external_constraints);
}
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, GlobalObservables global_observables)
{
   return buildUnbinnedConstrainedLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, {}, global_observables);
}
std::shared_ptr<RooAbsL> buildUnbinnedConstrainedLikelihood(RooAbsPdf* pdf, RooAbsData* data, std::string global_observables_tag)
{
   return buildUnbinnedConstrainedLikelihood(pdf, data, RooAbsL::Extended::Auto, {}, {}, {}, global_observables_tag);
}

}
}

