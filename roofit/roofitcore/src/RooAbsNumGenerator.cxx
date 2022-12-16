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

Class RooAbsNumGenerator is the abstract base class for MC event generator
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

#include <assert.h>

using namespace std;



////////////////////////////////////////////////////////////////////////////////
/// Initialize an accept-reject generator for the specified distribution function,
/// which must be non-negative but does not need to be normalized over the
/// variables to be generated, genVars. The function and its dependents are
/// cloned and so will not be disturbed during the generation process.

RooAbsNumGenerator::RooAbsNumGenerator(const RooAbsReal &func, const RooArgSet &genVars, bool verbose, const RooAbsReal* maxFuncVal) :
  _cloneSet(0), _funcClone(0), _funcMaxVal(maxFuncVal), _verbose(verbose), _funcValStore(0), _funcValPtr(0), _cache(0)
{
  // Clone the function and all nodes that it depends on so that this generator
  // is independent of any existing objects.
  RooArgSet nodes(func,func.GetName());
  _cloneSet= (RooArgSet*) nodes.snapshot(true);
  if (!_cloneSet) {
    oocoutE(nullptr, Generation) << "RooAbsNumGenerator::RooAbsNumGenerator(" << func.GetName() << ") Couldn't deep-clone function, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }

  // Find the clone in the snapshot list
  _funcClone = (RooAbsReal*)_cloneSet->find(func.GetName());


  // Check that each argument is fundamental, and separate them into
  // sets of categories and reals. Check that the area of the generating
  // space is finite.
  _isValid= true;
  const RooAbsArg *found = 0;
  for (RooAbsArg const* arg : genVars) {
    if(!arg->isFundamental()) {
      oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator: cannot generate values for derived \""
         << arg->GetName() << "\"" << endl;
      _isValid= false;
      continue;
    }
    // look for this argument in the generating function's dependents
    found= (const RooAbsArg*)_cloneSet->find(arg->GetName());
    if(found) {
      arg= found;
    } else {
      // clone any variables we generate that we haven't cloned already
      arg= _cloneSet->addClone(*arg);
    }
    assert(0 != arg);
    // is this argument a category or a real?
    const RooCategory *catVar= dynamic_cast<const RooCategory*>(arg);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(arg);
    if(0 != catVar) {
      _catVars.add(*catVar);
    }
    else if(0 != realVar) {
      if(realVar->hasMin() && realVar->hasMax()) {
   _realVars.add(*realVar);
      }
      else {
   oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator: cannot generate values for \""
           << realVar->GetName() << "\" with unbound range" << endl;
   _isValid= false;
      }
    }
    else {
      oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator" << ": cannot generate values for \""
         << arg->GetName() << "\" with unexpected type" << endl;
      _isValid= false;
    }
  }
  if(!_isValid) {
    oocoutE(nullptr, Generation) << func.GetName() << "::RooAbsNumGenerator" << ": constructor failed with errors" << endl;
    return;
  }

  // create a fundamental type for storing function values
  _funcValStore= dynamic_cast<RooRealVar*>(_funcClone->createFundamental());
  assert(0 != _funcValStore);

  // create a new dataset to cache trial events and function values
  RooArgSet cacheArgs(_catVars);
  cacheArgs.add(_realVars);
  cacheArgs.add(*_funcValStore);
  _cache= new RooDataSet("cache","Accept-Reject Event Cache",cacheArgs);
  assert(0 != _cache);

  // attach our function clone to the cache dataset
  const RooArgSet *cacheVars= _cache->get();
  assert(0 != cacheVars);
  _funcClone->recursiveRedirectServers(*cacheVars,false);

  // update ours sets of category and real args to refer to the cache dataset
  const RooArgSet *dataVars= _cache->get();
  _catVars.replace(*dataVars);
  _realVars.replace(*dataVars);

  // find the function value in the dataset
  _funcValPtr= (RooRealVar*)dataVars->find(_funcValStore->GetName());

}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsNumGenerator::~RooAbsNumGenerator()
{
  delete _cloneSet;
  delete _cache ;
  delete _funcValStore ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reattach original parameters to function clone

void RooAbsNumGenerator::attachParameters(const RooArgSet& vars)
{
  RooArgSet newParams(vars) ;
  newParams.remove(*_cache->get(),true,true) ;
  _funcClone->recursiveRedirectServers(newParams) ;
}
