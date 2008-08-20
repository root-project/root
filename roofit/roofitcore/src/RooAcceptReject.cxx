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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// Class RooAcceptReject is a generic toy monte carlo generator implement
// the accept/reject sampling technique on any positively valued function.
// The RooAcceptReject generator is used by the various generator context
// classes to take care of generation of observables for which p.d.fs
// do not define internal methods
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooAcceptReject.h"
#include "RooAcceptReject.h"
#include "RooAbsReal.h"
#include "RooCategory.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooRandom.h"
#include "RooErrorHandler.h"

#include "TString.h"
#include "TIterator.h"
#include "RooMsgService.h"
#include "TClass.h"

#include <assert.h>

ClassImp(RooAcceptReject)
  ;


//_____________________________________________________________________________
RooAcceptReject::RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, const RooAbsReal* maxFuncVal, Bool_t verbose) :
  TNamed(func), _cloneSet(0), _funcClone(0), _funcMaxVal(maxFuncVal), _verbose(verbose)
{
  // Initialize an accept-reject generator for the specified distribution function,
  // which must be non-negative but does not need to be normalized over the
  // variables to be generated, genVars. The function and its dependents are
  // cloned and so will not be disturbed during the generation process.

  // Clone the function and all nodes that it depends on so that this generator
  // is independent of any existing objects.
  RooArgSet nodes(func,func.GetName());
  _cloneSet= (RooArgSet*) nodes.snapshot(kTRUE);
  if (!_cloneSet) {
    coutE(Generation) << "RooAcceptReject::RooAcceptReject(" << GetName() << ") Couldn't deep-clone function, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }

  // Find the clone in the snapshot list
  _funcClone = (RooAbsReal*)_cloneSet->find(func.GetName());
  
  // Check that each argument is fundamental, and separate them into
  // sets of categories and reals. Check that the area of the generating
  // space is finite.
  _realSampleDim= 0;
  _catSampleMult= 1;
  _isValid= kTRUE;
  TIterator *iterator= genVars.createIterator();
  const RooAbsArg *found = 0;
  const RooAbsArg *arg   = 0;
  while((arg= (const RooAbsArg*)iterator->Next())) {
    if(!arg->isFundamental()) {
      coutE(Generation) << fName << "::" << ClassName() << ": cannot generate values for derived \""
			<< arg->GetName() << "\"" << endl;
      _isValid= kFALSE;
      continue;
    }
    // look for this argument in the generating function's dependents
    found= (const RooAbsArg*)_cloneSet->find(arg->GetName());
    if(found) {
      arg= found;
      const RooAbsCategory * cat = dynamic_cast<const RooAbsCategory*>(found) ;
      if (cat) {
	_catSampleMult *= cat->numTypes() ;
      } else {
	_realSampleDim++;
      }
    }
    else {
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
	coutE(Generation) << fName << "::" << ClassName() << ": cannot generate values for \""
			  << realVar->GetName() << "\" with unbound range" << endl;
	_isValid= kFALSE;
      }
    }
    else {
      coutE(Generation) << fName << "::" << ClassName() << ": cannot generate values for \""
			<< arg->GetName() << "\" with unexpected type" << endl;
      _isValid= kFALSE;
    }
  }
  delete iterator;
  if(!_isValid) {
    coutE(Generation) << fName << "::" << ClassName() << ": constructor failed with errors" << endl;
    return;
  }

  // calculate the minimum number of trials needed to estimate our integral and max value
  if (!_funcMaxVal) {
    if(_realSampleDim > _maxSampleDim) {
      _minTrials= _minTrialsArray[_maxSampleDim]*_catSampleMult;
      coutW(Generation) << fName << "::" << ClassName() << ": WARNING: generating " << _realSampleDim
			<< " variables with accept-reject may not be accurate" << endl;
    }
    else {
      _minTrials= _minTrialsArray[_realSampleDim]*_catSampleMult;
    }
  } else {
    // No trials needed if we know the maximum a priori
    _minTrials=0 ;
  }

  // print a verbose summary of our configuration, if requested
  if(_verbose) {
    coutI(Generation) << fName << "::" << ClassName() << ":" << endl
		      << "  Initializing accept-reject generator for" << endl << "    ";
    _funcClone->printStream(ccoutI(Generation),kName,kSingleLine);
    if (_funcMaxVal) {
      ccoutI(Generation) << "  Function maximum provided, no trial sampling performed" << endl ;
    } else {
      ccoutI(Generation) << "  Real sampling dimension is " << _realSampleDim << endl;
      ccoutI(Generation) << "  Category sampling multiplier is " << _catSampleMult << endl ;
      ccoutI(Generation) << "  Min sampling trials is " << _minTrials << endl;
    }
    if (_catVars.getSize()>0) {
      ccoutI(Generation) << "  Will generate category vars "<< _catVars << endl ;
    }
    if (_realVars.getSize()>0) {
      ccoutI(Generation) << "  Will generate real vars " << _realVars << endl ;
    }
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
  _funcClone->recursiveRedirectServers(*cacheVars,kFALSE);

  // update ours sets of category and real args to refer to the cache dataset
  const RooArgSet *dataVars= _cache->get();
  _catVars.replace(*dataVars);
  _realVars.replace(*dataVars);

  // find the function value in the dataset
  _funcValPtr= (RooRealVar*)dataVars->find(_funcValStore->GetName());

  // create iterators for the new sets
  _nextCatVar= _catVars.createIterator();
  _nextRealVar= _realVars.createIterator();
  assert(0 != _nextCatVar && 0 != _nextRealVar);

  // initialize our statistics
  _maxFuncVal= 0;
  _funcSum= 0;
  _totalEvents= 0;
  _eventsUsed= 0;
}



//_____________________________________________________________________________
RooAcceptReject::~RooAcceptReject() 
{
  // Destructor

  delete _cache ;
  delete _nextCatVar;
  delete _nextRealVar;
  delete _cloneSet;
  delete _funcValStore;
}



//_____________________________________________________________________________
void RooAcceptReject::attachParameters(const RooArgSet& vars) 
{
  // Reattach original parameters to function clone

  RooArgSet newParams(vars) ;
  newParams.remove(*_cache->get(),kTRUE,kTRUE) ;
  _funcClone->recursiveRedirectServers(newParams) ;
}



//_____________________________________________________________________________
const RooArgSet *RooAcceptReject::generateEvent(UInt_t remaining) 
{
  // Return a pointer to a generated event. The caller does not own the event and it
  // will be overwritten by a subsequent call. The input parameter 'remaining' should
  // contain your best guess at the total number of subsequent events you will request.

  // are we actually generating anything? (the cache always contains at least our function value)
  const RooArgSet *event= _cache->get();
  if(event->getSize() == 1) return event;

  if (!_funcMaxVal) {
    // Generation with empirical maximum determination

    // first generate enough events to get reasonable estimates for the integral and
    // maximum function value

    while(_totalEvents < _minTrials) {
      addEventToCache();

      // Limit cache size to 1M events
      if (_cache->numEntries()>1000000) {
	coutI(Generation) << "RooAcceptReject::generateEvent: resetting event cache" << endl ;
	_cache->reset() ;
	_eventsUsed = 0 ;
      }
    }
    
    event= 0;
    while(0 == event) {
      // Use any cached events first
      event= nextAcceptedEvent();
      if(event) break;
      // When we have used up the cache, start a new cache and add
      // some more events to it.      
      _cache->reset();
      _eventsUsed= 0;
      // Calculate how many more events to generate using our best estimate of our efficiency.
      // Always generate at least one more event so we don't get stuck.
      if(_totalEvents*_maxFuncVal <= 0) {
	coutE(Generation) << "RooAcceptReject::generateEvent: cannot estimate efficiency...giving up" << endl;
	return 0;
      }
      Double_t eff= _funcSum/(_totalEvents*_maxFuncVal);
      Int_t extra= 1 + (Int_t)(1.05*remaining/eff);
      cxcoutD(Generation) << "RooAcceptReject::generateEvent: adding " << extra << " events to the cache" << endl;
      Double_t oldMax(_maxFuncVal);
      while(extra--) addEventToCache();
      if((_maxFuncVal > oldMax)) {
	cxcoutD(Generation) << "RooAcceptReject::generateEvent: estimated function maximum increased from "
			  << oldMax << " to " << _maxFuncVal << endl;
      }
    }

    // Limit cache size to 1M events
    if (_eventsUsed>1000000) {
      _cache->reset() ;
      _eventsUsed = 0 ;
    }

  } else {
    // Generation with a priori maximum knowledge
    _maxFuncVal = _funcMaxVal->getVal() ;
    
    // Generate enough trials to produce a single accepted event
    event = 0 ;
    while(0==event) {
      addEventToCache() ;
      event = nextAcceptedEvent() ;
    }

  }
  return event;
}



//_____________________________________________________________________________
const RooArgSet *RooAcceptReject::nextAcceptedEvent() 
{
  // Scan through events in the cache which have not been used yet,
  // looking for the first accepted one which is added to the specified
  // container. Return a pointer to the accepted event, or else zero
  // if we use up the cache before we accept an event. The caller does
  // not own the event and it will be overwritten by a subsequent call.

  const RooArgSet *event = 0;
  while((event= _cache->get(_eventsUsed))) {    
    _eventsUsed++ ;
    // accept this cached event?
    Double_t r= RooRandom::uniform();
    if(r*_maxFuncVal > _funcValPtr->getVal()) continue;
    // copy this event into the output container
    if(_verbose && (_eventsUsed%1000==0)) {
      cerr << "RooAcceptReject: accepted event (used " << _eventsUsed << " of "
	   << _cache->numEntries() << " so far)" << endl;
    }
    break;
  }  
  return event;
}



//_____________________________________________________________________________
void RooAcceptReject::addEventToCache() 
{
  // Add a trial event to our cache and update our estimates
  // of the function maximum value and integral.

  // randomize each discrete argument
  _nextCatVar->Reset();
  RooCategory *cat = 0;
  while((cat= (RooCategory*)_nextCatVar->Next())) cat->randomize();

  // randomize each real argument
  _nextRealVar->Reset();
  RooRealVar *real = 0;
  while((real= (RooRealVar*)_nextRealVar->Next())) real->randomize();

  // calculate and store our function value at this new point
  Double_t val= _funcClone->getVal();
  _funcValPtr->setVal(val);

  // Update the estimated integral and maximum value. Increase our
  // maximum estimate slightly to give a safety margin with a
  // corresponding loss of efficiency.
  if(val > _maxFuncVal) _maxFuncVal= 1.05*val;
  _funcSum+= val;

  // fill a new entry in our cache dataset for this point
  _cache->fill();
  _totalEvents++;

  if (_verbose &&_totalEvents%10000==0) {
    cerr << "RooAcceptReject: generated " << _totalEvents << " events so far." << endl ;
  }

}

Double_t RooAcceptReject::getFuncMax() 
{
  // Empirically determine maximum value of function by taking a large number
  // of samples. The actual number depends on the number of dimensions in which
  // the sampling occurs

  // Generate the minimum required number of samples for a reliable maximum estimate
  while(_totalEvents < _minTrials) {
    addEventToCache();

    // Limit cache size to 1M events
    if (_cache->numEntries()>1000000) {
      coutI(Generation) << "RooAcceptReject::getFuncMax: resetting event cache" << endl ;
      _cache->reset() ;
      _eventsUsed = 0 ;
    }
  }  

  return _maxFuncVal ;
}


//_____________________________________________________________________________
void RooAcceptReject::printName(ostream& os) const 
{
  // Print name of the generator

  os << GetName() ;
}



//_____________________________________________________________________________
void RooAcceptReject::printTitle(ostream& os) const 
{
  // Print the title of the generator

  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooAcceptReject::printClassName(ostream& os) const 
{
  // Print the class name of the generator

  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void RooAcceptReject::printArgs(ostream& os) const 
{
  // Print the arguments of the generator

  os << "[ function=" << _funcClone->GetName() << " catobs=" << _catVars << " realobs=" << _realVars << " ]" ;
}


const UInt_t RooAcceptReject::_minTrialsArray[] = { 100,1000,100000,10000000 };
const UInt_t RooAcceptReject::_maxSampleDim = (sizeof(_minTrialsArray) / sizeof(int)) - 1;
