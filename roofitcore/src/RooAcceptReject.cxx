/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAcceptReject.cc,v 1.19 2001/11/06 18:32:59 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// A class description belongs here...

//#include "BaBar/BaBar.hh"

#include "RooFitCore/RooAcceptReject.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRandom.hh"

#include "TString.h"
#include "TIterator.h"

#include <assert.h>

ClassImp(RooAcceptReject)
  ;

static const char rcsid[] =
"$Id: RooAcceptReject.cc,v 1.19 2001/11/06 18:32:59 verkerke Exp $";

RooAcceptReject::RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, const RooAbsReal* maxFuncVal, Bool_t verbose) :
  TNamed(func), _cloneSet(0), _funcClone(0), _verbose(verbose), _funcMaxVal(maxFuncVal)
{
  // Initialize an accept-reject generator for the specified distribution function,
  // which must be non-negative but does not need to be normalized over the
  // variables to be generated, genVars. The function and its dependents are
  // cloned and so will not be disturbed during the generation process.

  // Clone the function and all nodes that it depends on so that this generator
  // is independent of any existing objects.
  RooArgSet nodes(func,func.GetName());
  _cloneSet= (RooArgSet*) nodes.snapshot(kTRUE);

  // Find the clone in the snapshot list
  _funcClone = (RooAbsReal*)_cloneSet->find(func.GetName());
  
  // Check that each argument is fundamental, and separate them into
  // sets of categories and reals. Check that the area of the generating
  // space is finite.
  _realSampleDim= 0;
  _catSampleMult= 1;
  _isValid= kTRUE;
  TIterator *iterator= genVars.createIterator();
  const RooAbsArg *found(0),*arg(0);
  while(arg= (const RooAbsArg*)iterator->Next()) {
    if(arg->isDerived()) {
      cout << fName << "::" << ClassName() << ": cannot generate values for derived \""
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
      if(realVar->hasFitMin() && realVar->hasFitMax()) {
	_realVars.add(*realVar);
      }
      else {
	cout << fName << "::" << ClassName() << ": cannot generate values for \""
	     << realVar->GetName() << "\" with unbound range" << endl;
	_isValid= kFALSE;
      }
    }
    else {
      cout << fName << "::" << ClassName() << ": cannot generate values for \""
	   << arg->GetName() << "\" with unexpected type" << endl;
      _isValid= kFALSE;
    }
  }
  delete iterator;
  if(!_isValid) {
    cout << fName << "::" << ClassName() << ": constructor failed with errors" << endl;
    return;
  }

  // calculate the minimum number of trials needed to estimate our integral and max value
  if (!_funcMaxVal) {
    if(_realSampleDim > _maxSampleDim) {
      _minTrials= _minTrialsArray[_maxSampleDim]*_catSampleMult;
      cout << fName << "::" << ClassName() << ": WARNING: generating " << _realSampleDim
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
    cout << fName << "::" << ClassName() << ":" << endl
	 << "  Initializing accept-reject generator for" << endl << "    ";
    _funcClone->Print();
    if (_funcMaxVal) {
      cout << "  Function maximum provided, no trial sampling performed" << endl ;
    } else {
      cout << "  Real sampling dimension is " << _realSampleDim << endl;
      cout << "  Category sampling multiplier is " << _catSampleMult << endl ;
      cout << "  Min sampling trials is " << _minTrials << endl;
    }
    cout << "  Will generate category vars ";
    TString indent("  ");
    _catVars.printToStream(cout,Standard,indent);
    cout << "  Will generate real vars ";
    _realVars.printToStream(cout,Standard,indent);
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

RooAcceptReject::~RooAcceptReject() {
  delete _cache ;
  delete _nextCatVar;
  delete _nextRealVar;
  delete _cloneSet;
  delete _funcValStore;
}

void RooAcceptReject::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  oneLinePrint(os,*this);
}



void RooAcceptReject::attachParameters(const RooArgSet& vars) 
{
  // Reattach original parameters to function clone
  RooArgSet newParams(vars) ;
  newParams.remove(*_cache->get(),kTRUE,kTRUE) ;
  _funcClone->recursiveRedirectServers(newParams) ;
}


const RooArgSet *RooAcceptReject::generateEvent(UInt_t remaining) {
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
	cout << "RooAcceptReject::generateEvent: resetting event cache" << endl ;
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
	cout << "RooAcceptReject::generateEvent: cannot estimate efficiency...giving up" << endl;
	return 0;
      }
      Double_t eff= _funcSum/(_totalEvents*_maxFuncVal);
      Int_t extra= 1 + (Int_t)(1.05*remaining/eff);
      if(_verbose) {
	cout << "RooAcceptReject::generateEvent: adding " << extra << " events to the cache" << endl;
      }
      Double_t oldMax(_maxFuncVal);
      while(extra--) addEventToCache();
      if(_verbose && (_maxFuncVal > oldMax)) {
	cout << "RooAcceptReject::generateEvent: estimated function maximum increased from "
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

const RooArgSet *RooAcceptReject::nextAcceptedEvent() {
  // Scan through events in the cache which have not been used yet,
  // looking for the first accepted one which is added to the specified
  // container. Return a pointer to the accepted event, or else zero
  // if we use up the cache before we accept an event. The caller does
  // not own the event and it will be overwritten by a subsequent call.

  const RooArgSet *event(0);
  while(event= _cache->get(_eventsUsed)) {    
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

void RooAcceptReject::addEventToCache() {
  // Add a trial event to our cache and update our estimates
  // of the function maximum value and integral.

  // randomize each discrete argument
  _nextCatVar->Reset();
  RooCategory *cat(0);
  while(cat= (RooCategory*)_nextCatVar->Next()) cat->randomize();

  // randomize each real argument
  _nextRealVar->Reset();
  RooRealVar *real(0);
  while(real= (RooRealVar*)_nextRealVar->Next()) real->randomize();

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
  // Generate the minimum required number of samples for a reliable maximum estimate
  while(_totalEvents < _minTrials) addEventToCache();
  
  return _maxFuncVal ;
}


const int RooAcceptReject::_maxSampleDim= 3,
  RooAcceptReject::_minTrialsArray[]= { 0,1000,100000,10000000 };
