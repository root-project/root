/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooConstraintSum.cxx
\class RooConstraintSum
\ingroup Roofitcore

RooConstraintSum calculates the sum of the -(log) likelihoods of
a set of RooAbsPfs that represent constraint functions. This class
is used to calculate the composite -log(L) of constraints to be
added to the regular -log(L) in RooAbsPdf::fitTo() with Constrain(..)
arguments.
**/


#include "RooConstraintSum.h"
#include "RooAbsData.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooHelpers.h"
#include "RooWorkspace.h"
#include "RooAbsRealLValue.h"
#include "RooAbsCategoryLValue.h"

#include <memory>

ClassImp(RooConstraintSum);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with set of constraint p.d.f.s. All elements in constraintSet must inherit from RooAbsPdf.

RooConstraintSum::RooConstraintSum(const char* name, const char* title, const RooArgSet& constraintSet, const RooArgSet& normSet, bool takeGlobalObservablesFromData) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _paramSet("paramSet","Set of parameters",this),
  _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
  for (const auto comp : constraintSet) {
    if (!dynamic_cast<RooAbsPdf*>(comp)) {
      coutE(InputArguments) << "RooConstraintSum::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
                            << " is not of type RooAbsPdf" << std::endl ;
      RooErrorHandler::softAbort() ;
    }
    _set1.add(*comp) ;
  }

  _paramSet.add(normSet) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

RooConstraintSum::RooConstraintSum(const RooConstraintSum& other, const char* name) :
  RooAbsReal(other, name), 
  _set1("set1",this,other._set1),
  _paramSet("paramSet",this,other._paramSet),
  _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return sum of -log of constraint p.d.f.s.

Double_t RooConstraintSum::evaluate() const 
{
  Double_t sum(0);

  for (const auto comp : _set1) {
    sum -= static_cast<RooAbsPdf*>(comp)->getLogVal(&_paramSet);
  }
  
  return sum;
}


////////////////////////////////////////////////////////////////////////////////
/// Replace the variables in this RooConstraintSum with the global observables
/// in the dataset if they match by name. This function will do nothing if this
/// RooConstraintSum is configured to not use the global observables stored in
/// datasets.
bool RooConstraintSum::setData(RooAbsData const& data, bool /*cloneData=true*/) {
  if(_takeGlobalObservablesFromData && data.getGlobalObservables()) {
    this->recursiveRedirectServers(*data.getGlobalObservables()) ;
  }
  return true;
}


namespace {

std::unique_ptr<RooArgSet> getGlobalObservables(
        RooAbsPdf const& pdf, RooArgSet const* globalObservables, const char* globalObservablesTag)
{

  if(globalObservables && globalObservablesTag) {
    // error!
    std::string errMsg = "RooAbsPdf::fitTo: GlobalObservables and GlobalObservablesTag options mutually exclusive!";
    oocoutE(&pdf, Minimization) << errMsg << std::endl;
    throw std::invalid_argument(errMsg);
  }
  if(globalObservables) {
    // pass-throught of global observables
    return std::make_unique<RooArgSet>(*globalObservables);
  }

  if(globalObservablesTag) {
    oocoutI(&pdf, Minimization) << "User-defined specification of global observables definition with tag named '"
                                <<  globalObservablesTag << "'" << std::endl;
  } else {
    // Neither GlobalObservables nor GlobalObservablesTag has been processed -
    // try if a default tag is defined in the head node Check if head not
    // specifies default global observable tag
    if(auto defaultGlobalObservablesTag = pdf.getStringAttribute("DefaultGlobalObservablesTag")) {
      oocoutI(&pdf, Minimization) << "p.d.f. provides built-in specification of global observables definition "
                                  << "with tag named '" <<  defaultGlobalObservablesTag << "'" << std::endl;
      globalObservablesTag = defaultGlobalObservablesTag;
    }
  }

  if(globalObservablesTag) {
    std::unique_ptr<RooArgSet> allVars{pdf.getVariables()} ;
    return std::unique_ptr<RooArgSet>{static_cast<RooArgSet*>(allVars->selectByAttrib(globalObservablesTag, true))};
  }

  // no global observables specified
  return nullptr;
}


RooArgSet const* tryToGetConstraintSetFromWorkspace(
        RooAbsPdf const& pdf, RooWorkspace * workspace, std::string const& constraintSetCacheName) {
  if(!workspace) return nullptr;

  if(workspace->set(constraintSetCacheName.c_str())) {
    // retrieve from cache
    const RooArgSet *constr = workspace->set(constraintSetCacheName.c_str());
    oocoutI(&pdf, Minimization)
        << "createConstraintTerm picked up cached constraints from workspace with " << constr->size()
        << " entries" << std::endl;
    return constr;
  }
  return nullptr;
}


} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Create the parameter constraint sum to add to the negative log-likelihood.
/// Returns a `nullptr` if the parameters are unconstrained.
/// \param[in] name Name of the created RooConstraintSum object.
/// \param[in] pdf The pdf model whose parameters should be constrained.
///            Constraint terms will be extracted from RooProdPdf instances
///            that are servers of the pdf (internal constraints).
/// \param[in] data Dataset used in the fit with the constraint sum. It is
///            used to figure out which are the observables and also to get the
///            global observables definition and values if they are stored in
///            the dataset.
/// \param[in] constraints Set of parameters to constrain. If `nullptr`, all
///            parameters will be considered.
/// \param[in] externalConstraints Set of constraint terms that are not
///            embedded in the pdf (external constraints). 
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
/// \param[in] workspace RooWorkspace to cache the set of constraints.
std::unique_ptr<RooAbsReal> RooConstraintSum::createConstraintTerm(
        std::string const& name,
        RooAbsPdf const& pdf,
        RooAbsData const& data,
        RooArgSet const* constrainedParameters,
        RooArgSet const* externalConstraints,
        RooArgSet const* globalObservables,
        const char* globalObservablesTag,
        bool takeGlobalObservablesFromData,
        bool cloneConstraints,
        RooWorkspace * workspace)
{
  RooArgSet const& observables = *data.get();

  bool doStripDisconnected = false ;

  // If no explicit list of parameters to be constrained is specified apply default algorithm
  // All terms of RooProdPdfs that do not contain observables and share a parameters with one or more
  // terms that do contain observables are added as constrainedParameters.
  RooArgSet cPars;
  if(constrainedParameters) {
    cPars.add(*constrainedParameters);
  } else {
    pdf.getParameters(&observables,cPars,false);
    doStripDisconnected = true;
  }

  // Collect internal and external constraint specifications
  RooArgSet allConstraints ;

  auto observableNames = RooHelpers::getColonSeparatedNameString(observables);
  auto constraintSetCacheName = std::string("CACHE_CONSTR_OF_PDF_") + pdf.GetName() + "_FOR_OBS_" +  observableNames;

  if (RooArgSet const* constr = tryToGetConstraintSetFromWorkspace(pdf, workspace, constraintSetCacheName)) {
    allConstraints.add(*constr);
  } else {

     if (!cPars.empty()) {
        std::unique_ptr<RooArgSet> internalConstraints{pdf.getAllConstraints(observables, cPars, doStripDisconnected)};
        allConstraints.add(*internalConstraints);
     }
     if (externalConstraints) {
        allConstraints.add(*externalConstraints);
     }

     // write to cache
     if (workspace) {
        oocoutI(&pdf, Minimization)
            << "createConstraintTerm: caching constraint set under name "
            << constraintSetCacheName << " with " << allConstraints.size() << " entries" << std::endl;
        workspace->defineSetInternal(constraintSetCacheName.c_str(), allConstraints);
     }
  }

  if (!allConstraints.empty()) {

    oocoutI(&pdf, Minimization) << " Including the following constraint terms in minimization: "
                                << allConstraints << std::endl ;

    // Identify global observables in the model.
    auto glObs = getGlobalObservables(pdf, globalObservables, globalObservablesTag);
    if(data.getGlobalObservables() && takeGlobalObservablesFromData) {
      if(!glObs) {
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
        oocoutI(&pdf, Minimization)
            << "The following global observables have been defined: " << *glObs
            << "," << " with the values of " << globalsFromDataset
            << " obtained from the dataset and the other values from the model." << std::endl;
      }
    } else if(glObs) {
      oocoutI(&pdf, Minimization)
          << "The following global observables have been defined and their values are taken from the model: "
          << *glObs << std::endl;
      // in this case we don;t take global observables from data
      takeGlobalObservablesFromData = false;
    } else {
       if (!glObs)
          oocoutI(&pdf, Minimization)
             << "The global observables are not defined , normalize constraints with respect to the parameters " << cPars
             << std::endl;
       takeGlobalObservablesFromData = false;
    }

    // The constraint terms need to be cloned, because the global observables
    // might be changed to have the same values as stored in data.
    if (cloneConstraints) {
       RooConstraintSum constraintTerm{name.c_str(), "nllCons", allConstraints, glObs ? *glObs : cPars,
                                       takeGlobalObservablesFromData};
       std::unique_ptr<RooAbsReal> constraintTermClone{static_cast<RooAbsReal *>(constraintTerm.cloneTree())};

       // The parameters that are not connected to global observables from data
       // need to be redirected to the original args to get the changes made by
       // the minimizer. This excludes the global observables, where we take the
       // clones with the values set to the values from the dataset if available.
       RooArgSet allOriginalParams;
       constraintTerm.getParameters(nullptr, allOriginalParams);
       constraintTermClone->recursiveRedirectServers(allOriginalParams);

       // Redirect the global observables to the ones from the dataset if applicable.
       static_cast<RooConstraintSum *>(constraintTermClone.get())->setData(data, false);

       // The computation graph for the constraints is very small, no need to do
       // the tracking of clean and dirty nodes here.
       constraintTermClone->setOperMode(RooAbsArg::ADirty);

       return constraintTermClone;
    }
    // case we do not clone constraints (e.g. when using new Driver)
    else {
       // when we don't clone we need that global observables are not from data
       if (takeGlobalObservablesFromData) {
          oocoutE(&pdf, InputArguments) << "RooAbsPdf::Fit: Batch mode does not support yet GlobalObservable from data, use model as GlobalObservableSource " << std::endl;
          throw std::invalid_argument("Invalid arguments for GlobalObservables in batch mode");
       }
       std::unique_ptr<RooAbsReal> constraintTerm(
          new RooConstraintSum(name.c_str(), "nllCons", allConstraints, glObs ? *glObs : cPars, false));
       return constraintTerm;
    }
  }

  // no constraints
  return nullptr;
}
