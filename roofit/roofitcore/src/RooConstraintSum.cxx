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
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooHelpers.h"
#include "RooWorkspace.h"

#include <memory>

ClassImp(RooConstraintSum);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with set of constraint p.d.f.s. All elements in constraintSet must inherit from RooAbsPdf.

RooConstraintSum::RooConstraintSum(const char* name, const char* title, const RooArgSet& constraintSet, const RooArgSet& normSet) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _paramSet("paramSet","Set of parameters",this)
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
  _paramSet("paramSet",this,other._paramSet)
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
/// \param[in] observables Observable variables (used to determine which model
///            variables are parameters).
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
/// \param[in] workspace RooWorkspace to cache the set of constraints.
std::unique_ptr<RooAbsReal> RooConstraintSum::createConstraintTerm(
        std::string const& name,
        RooAbsPdf const& pdf,
        RooArgSet const& observables,
        RooArgSet const* constrainedParameters,
        RooArgSet const* externalConstraints,
        RooArgSet const* globalObservables,
        const char* globalObservablesTag,
        RooWorkspace * workspace)
{
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

  auto glObs = getGlobalObservables(pdf, globalObservables, globalObservablesTag);

  if (!allConstraints.empty()) {

    oocoutI(&pdf, Minimization) << " Including the following constraint terms in minimization: " << allConstraints << std::endl ;
    if (glObs) {
      oocoutI(&pdf, Minimization) << "The following global observables have been defined: " << *glObs << std::endl ;
    }
    auto constraintTerm = std::make_unique<RooConstraintSum>(name.c_str(),"nllCons",allConstraints,glObs ? *glObs : cPars) ;
    constraintTerm->setOperMode(RooAbsArg::ADirty) ;
    return constraintTerm;
  }

  // no constraints
  return nullptr;
}
