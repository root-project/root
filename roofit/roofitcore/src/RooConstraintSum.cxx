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

Calculates the sum of the -(log) likelihoods of
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
#include "RooAbsCategoryLValue.h"

ClassImp(RooConstraintSum);


////////////////////////////////////////////////////////////////////////////////
/// Constructor with set of constraint p.d.f.s. All elements in constraintSet must inherit from RooAbsPdf.

RooConstraintSum::RooConstraintSum(const char* name, const char* title, const RooArgSet& constraintSet, const RooArgSet& normSet, bool takeGlobalObservablesFromData) :
  RooAbsReal(name, title),
  _set1("set1","First set of components",this),
  _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
  _set1.addTyped<RooAbsPdf>(constraintSet);
  _paramSet.add(normSet) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

RooConstraintSum::RooConstraintSum(const RooConstraintSum& other, const char* name) :
  RooAbsReal(other, name),
  _set1("set1",this,other._set1),
  _paramSet(other._paramSet),
  _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData}
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return sum of -log of constraint p.d.f.s.

double RooConstraintSum::evaluate() const
{
  double sum(0);

  for (const auto comp : _set1) {
    sum -= static_cast<RooAbsPdf*>(comp)->getLogVal(&_paramSet);
  }

  return sum;
}

void RooConstraintSum::doEval(RooFit::EvalContext &ctx) const
{
   double sum(0);

   for (const auto comp : _set1) {
      sum -= std::log(ctx.at(comp)[0]);
   }

   ctx.output()[0] = sum;
}

std::unique_ptr<RooAbsArg> RooConstraintSum::compileForNormSet(RooArgSet const & /*normSet*/, RooFit::Detail::CompileContext & ctx) const
{
   std::unique_ptr<RooAbsReal> newArg{static_cast<RooAbsReal*>(this->Clone())};

   for (const auto server : newArg->servers()) {
      RooArgSet nset;
      server->getObservables(&_paramSet, nset);
      ctx.compileServer(*server, *newArg, nset);
   }

   return newArg;
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
