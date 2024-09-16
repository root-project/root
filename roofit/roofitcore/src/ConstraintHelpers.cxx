/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "ConstraintHelpers.h"

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooConstraintSum.h>
#include <RooMsgService.h>

#include "RooFitImplHelpers.h"

namespace {

std::unique_ptr<RooArgSet>
getGlobalObservables(RooAbsPdf const &pdf, RooArgSet const *globalObservables, const char *globalObservablesTag)
{

   if (globalObservables && globalObservablesTag) {
      // error!
      std::string errMsg = "RooAbsPdf::fitTo: GlobalObservables and GlobalObservablesTag options mutually exclusive!";
      oocoutE(&pdf, Minimization) << errMsg << std::endl;
      throw std::invalid_argument(errMsg);
   }
   if (globalObservables) {
      // pass-through of global observables
      return std::make_unique<RooArgSet>(*globalObservables);
   }

   if (globalObservablesTag) {
      oocoutI(&pdf, Minimization) << "User-defined specification of global observables definition with tag named '"
                                  << globalObservablesTag << "'" << std::endl;
   } else {
      // Neither GlobalObservables nor GlobalObservablesTag has been processed -
      // try if a default tag is defined in the head node Check if head not
      // specifies default global observable tag
      if (auto defaultGlobalObservablesTag = pdf.getStringAttribute("DefaultGlobalObservablesTag")) {
         oocoutI(&pdf, Minimization) << "p.d.f. provides built-in specification of global observables definition "
                                     << "with tag named '" << defaultGlobalObservablesTag << "'" << std::endl;
         globalObservablesTag = defaultGlobalObservablesTag;
      }
   }

   if (globalObservablesTag) {
      std::unique_ptr<RooArgSet> allVars{pdf.getVariables()};
      return std::unique_ptr<RooArgSet>{static_cast<RooArgSet *>(allVars->selectByAttrib(globalObservablesTag, true))};
   }

   // no global observables specified
   return nullptr;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Create the parameter constraint sum to add to the negative log-likelihood.
/// \return If there are constraints, returns a pointer to the constraint NLL.
///         Returns a `nullptr` if the parameters are unconstrained.
/// \param[in] name Name of the created RooConstraintSum object.
/// \param[in] pdf The PDF model whose parameters should be constrained.
///            Constraint terms will be extracted from RooProdPdf instances
///            that are servers of the PDF (internal constraints).
/// \param[in] data Dataset used in the fit with the constraint sum. It is
///            used to figure out which are the observables and also to get the
///            global observables definition and values if they are stored in
///            the dataset.
/// \param[in] constrainedParameters Set of parameters to constrain. If `nullptr`, all
///            parameters will be considered.
/// \param[in] externalConstraints Set of constraint terms that are not
///            embedded in the PDF (external constraints).
/// \param[in] globalObservables The normalization set for the constraint terms.
///            If it is `nullptr`, the set of all constrained parameters will
///            be used as the normalization set.
/// \param[in] globalObservablesTag Alternative to define the normalization set
///            for the constraint terms. All constrained parameters that have
///            the attribute with the tag defined by `globalObservablesTag` are
///            used. The `globalObservables` and `globalObservablesTag`
///            parameters are mutually exclusive, meaning at least one of them
///            has to be `nullptr`.
/// \param[in] takeGlobalObservablesFromData If the dataset should be used to automatically
///            define the set of global observables. If this is the case and the
///            set of global observables is still defined manually with the
///            `globalObservables` or `globalObservablesTag` parameters, the
///            values of all global observables that are not stored in the
///            dataset are taken from the model.
std::unique_ptr<RooAbsReal> createConstraintTerm(std::string const &name, RooAbsPdf const &pdf, RooAbsData const &data,
                                                 RooArgSet const *constrainedParameters,
                                                 RooArgSet const *externalConstraints,
                                                 RooArgSet const *globalObservables, const char *globalObservablesTag,
                                                 bool takeGlobalObservablesFromData)
{
   RooArgSet const &observables = *data.get();

   bool doStripDisconnected = false;

   // If no explicit list of parameters to be constrained is specified apply default algorithm
   // All terms of RooProdPdfs that do not contain observables and share a parameters with one or more
   // terms that do contain observables are added as constrainedParameters.
   RooArgSet cPars;
   if (constrainedParameters) {
      cPars.add(*constrainedParameters);
   } else {
      pdf.getParameters(&observables, cPars, false);
      doStripDisconnected = true;
   }

   // Collect internal and external constraint specifications
   RooArgSet allConstraints;

   auto observableNames = RooHelpers::getColonSeparatedNameString(observables);
   auto constraintSetCacheName = std::string("CACHE_CONSTR_OF_PDF_") + pdf.GetName() + "_FOR_OBS_" + observableNames;

   if (!cPars.empty()) {
      std::unique_ptr<RooArgSet> internalConstraints{
         pdf.getAllConstraints(observables, cPars, doStripDisconnected)};
      allConstraints.add(*internalConstraints);
   }
   if (externalConstraints) {
      allConstraints.add(*externalConstraints);
   }

   if (!allConstraints.empty()) {

      oocoutI(&pdf, Minimization) << " Including the following constraint terms in minimization: " << allConstraints
                                  << std::endl;

      // Identify global observables in the model.
      auto glObs = getGlobalObservables(pdf, globalObservables, globalObservablesTag);
      if (data.getGlobalObservables() && takeGlobalObservablesFromData) {
         if (!glObs) {
            // There were no global observables specified, but there are some in the
            // dataset. We will just take them from the dataset.
            oocoutI(&pdf, Minimization)
               << "The following global observables have been automatically defined according to the dataset "
               << "which also provides their values: " << *data.getGlobalObservables() << std::endl;
            glObs = std::make_unique<RooArgSet>(*data.getGlobalObservables());
         } else {
            // There are global observables specified by the user and also some in
            // the dataset.
            RooArgSet globalsFromDataset;
            data.getGlobalObservables()->selectCommon(*glObs, globalsFromDataset);
            oocoutI(&pdf, Minimization) << "The following global observables have been defined: " << *glObs << ","
                                        << " with the values of " << globalsFromDataset
                                        << " obtained from the dataset and the other values from the model."
                                        << std::endl;
         }
      } else if (glObs) {
         oocoutI(&pdf, Minimization)
            << "The following global observables have been defined and their values are taken from the model: "
            << *glObs << std::endl;
         // In this case we don't take global observables from data
         takeGlobalObservablesFromData = false;
      } else {
         if (!glObs) {
            oocoutI(&pdf, Minimization)
               << "The global observables are not defined , normalize constraints with respect to the parameters "
               << cPars << std::endl;
         }
         takeGlobalObservablesFromData = false;
      }

      return std::make_unique<RooConstraintSum>(name.c_str(), "nllCons", allConstraints, glObs ? *glObs : cPars,
                                                takeGlobalObservablesFromData);
   }

   // no constraints
   return nullptr;
}
