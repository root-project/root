/// \cond ROOFIT_INTERNAL

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
\file RooAbsNumGenerator.cxx
\class RooAbsNumGenerator
\ingroup Roofitcore

Abstract base class for MC event generator
implementations like RooAcceptReject and RooFoam
**/

#include "Riostream.h"

#include "RooAbsNumGenerator.h"
#include "RooAbsReal.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooRandom.h"
#include "RooErrorHandler.h"

#include "RooMsgService.h"
#include "RooRealBinding.h"

#include <cassert>

using std::endl;

RooAbsNumGenerator::RooAbsNumGenerator() = default;


////////////////////////////////////////////////////////////////////////////////
/// Initialize an accept-reject generator for the specified distribution function,
/// which must be non-negative but does not need to be normalized over the
/// variables to be generated, genVars. The function and its dependents are
/// cloned and so will not be disturbed during the generation process.

RooAbsNumGenerator::RooAbsNumGenerator(const RooAbsReal &func, const RooArgSet &genVars, bool verbose, const RooAbsReal* maxFuncVal) :
  _funcMaxVal(maxFuncVal), _verbose(verbose)
{
  // Clone the function and all nodes that it depends on so that this generator
  // is independent of any existing objects.
  RooArgSet nodes(func,func.GetName());
  if (nodes.snapshot(_cloneSet, true)) {
    oocoutE(nullptr, Generation) << "RooAbsNumGenerator::RooAbsNumGenerator(" << func.GetName() << ") Couldn't deep-clone function, abort," << std::endl ;
    RooErrorHandler::softAbort() ;
  }

  // Find the clone in the snapshot list
  _funcClone = static_cast<RooAbsReal*>(_cloneSet.find(func.GetName()));


  // Check that each argument is fundamental, and separate them into
  // sets of categories and reals. Check that the area of the generating
  // space is finite.
  _isValid= true;
  const RooAbsArg *found = nullptr;
  for (RooAbsArg const* arg : genVars) {
    if(!arg->isFundamental()) {
      oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator: cannot generate values for derived \""
         << arg->GetName() << "\"" << std::endl;
      _isValid= false;
      continue;
    }
    // look for this argument in the generating function's dependents
    found= (const RooAbsArg*)_cloneSet.find(arg->GetName());
    if(found) {
      arg= found;
    } else {
      // clone any variables we generate that we haven't cloned already
      arg= _cloneSet.addClone(*arg);
    }
    assert(nullptr != arg);
    // is this argument a category or a real?
    const RooCategory *catVar= dynamic_cast<const RooCategory*>(arg);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(arg);
    if(nullptr != catVar) {
      _catVars.add(*catVar);
    }
    else if(nullptr != realVar) {
      if(realVar->hasMin() && realVar->hasMax()) {
   _realVars.add(*realVar);
      }
      else {
   oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator: cannot generate values for \""
           << realVar->GetName() << "\" with unbound range" << std::endl;
   _isValid= false;
      }
    }
    else {
      oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator" << ": cannot generate values for \""
         << arg->GetName() << "\" with unexpected type" << std::endl;
      _isValid= false;
    }
  }
  if(!_isValid) {
    oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator" << ": constructor failed with errors" << std::endl;
    return;
  }

  // create a fundamental type for storing function values
  _funcValStore= std::unique_ptr<RooAbsArg>{_funcClone->createFundamental()};

  // create a new dataset to cache trial events and function values
  RooArgSet cacheArgs(_catVars);
  cacheArgs.add(_realVars);
  cacheArgs.add(*_funcValStore);
  _cache= std::make_unique<RooDataSet>("cache","Accept-Reject Event Cache",cacheArgs);

  // attach our function clone to the cache dataset
  const RooArgSet *cacheVars= _cache->get();
  assert(nullptr != cacheVars);
  _funcClone->recursiveRedirectServers(*cacheVars,false);

  // update ours sets of category and real args to refer to the cache dataset
  const RooArgSet *dataVars= _cache->get();
  _catVars.replace(*dataVars);
  _realVars.replace(*dataVars);

  // find the function value in the dataset
  _funcValPtr= static_cast<RooRealVar*>(dataVars->find(_funcValStore->GetName()));

}


RooAbsNumGenerator::~RooAbsNumGenerator() = default;


////////////////////////////////////////////////////////////////////////////////
/// Reattach original parameters to function clone

void RooAbsNumGenerator::attachParameters(const RooArgSet& vars)
{
  RooArgSet newParams(vars) ;
  newParams.remove(*_cache->get(),true,true) ;
  _funcClone->recursiveRedirectServers(newParams) ;
}

/// \endcond
