/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAcceptReject.cc,v 1.5 2001/08/01 21:30:15 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
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
"$Id: RooAcceptReject.cc,v 1.5 2001/08/01 21:30:15 david Exp $";

RooAcceptReject::RooAcceptReject(const RooAbsReal &func, const RooArgSet &genVars, Bool_t verbose) :
  TNamed(func), _cloneSet(0), _funcClone(0), _verbose(verbose)
{
  // Initialize an accept-reject generator for the specified distribution function,
  // which must be non-negative but does not need to be normalized over the
  // variables to be generated, genVars. The function and its dependents are not
  // cloned and so will generally be disturbed during the generation process.

  // Clone the function and all nodes that it depends on so that this generator
  // is independent of any existing objects.
  RooArgSet nodes(func,func.GetName());
  _cloneSet= nodes.snapshot(kTRUE);

  // Find the clone in the snapshot list
  _funcClone = (RooAbsReal*)_cloneSet->FindObject(func.GetName());
  
  // Check that each argument is fundamental, and separate them into
  // sets of categories and reals. Check that the area of the generating
  // space is finite.
  _sampleDim= 0;
  _isValid= kTRUE;
  TIterator *iterator= genVars.MakeIterator();
  const RooAbsArg *found(0),*arg(0);
  while(arg= (const RooAbsArg*)iterator->Next()) {
    if(arg->isDerived()) {
      cout << fName << "::" << ClassName() << ": cannot generate values for derived \""
	   << arg->GetName() << "\"" << endl;
      _isValid= kFALSE;
      continue;
    }
    // look for this argument in the generating function's dependents
    found= (const RooAbsArg*)_cloneSet->FindObject(arg->GetName());
    if(found) {
      arg= found;
      _sampleDim++;
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
  if(_sampleDim > _maxSampleDim) {
    _minTrials= _minTrialsArray[_maxSampleDim];
    cout << fName << "::" << ClassName() << ": WARNING: generating " << _sampleDim
	 << " variables with accept-reject may not be accurate" << endl;
  }
  else {
    _minTrials= _minTrialsArray[_sampleDim];
  }
  // print a verbose summary of our configuration, if requested
  if(_verbose) {
    cout << fName << "::" << ClassName() << ":" << endl
	 << "  Initializing accept-reject generator for" << endl << "    ";
    _funcClone->Print();
    cout << "  Sampling dimension is " << _sampleDim << endl;
    cout << "  Min sampling trials is " << _minTrials << endl;
    cout << "  Will generate category vars ";
    TString indent("  ");
    _catVars.printToStream(cout,Standard,indent);
    cout << "  Will generate real vars ";
    _realVars.printToStream(cout,Standard,indent);
  }

  // create a fundamental type for storing function values
  _funcVal= dynamic_cast<RooRealVar*>(_funcClone->createFundamental());
  assert(0 != _funcVal);

  // initialize our statistics
  _maxFuncVal= 0;
  _funcSum= 0;
  _totalEvents= 0;
}

RooAcceptReject::~RooAcceptReject() {
  delete _cloneSet;
  delete _funcVal;
}

void RooAcceptReject::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  oneLinePrint(os,*this);
}

void RooAcceptReject::generateEvents(Int_t nEvents, RooDataSet &container) {
  // Fill the dataset provided with nEvents generated entries.

  // create a new dataset to cache trial events and function values
  RooArgSet cacheArgs(_catVars);
  cacheArgs.add(_realVars);
  cacheArgs.add(*_funcVal);
  RooDataSet cache("cache","Accept-Reject Event Cache",cacheArgs);
  _eventsUsed= 0;

  // attach our function clone to this dataset
  _funcClone->recursiveRedirectServers(*cache.get(),kFALSE);

  // create new sets of category and real args that refer to the new dataset
  const RooArgSet *dataVars= cache.get();
  RooArgSet catVars(_catVars),realVars(_realVars);
  catVars.replace(*dataVars);
  realVars.replace(*dataVars);

  // find the function value in the dataset
  RooRealVar *funcVal= (RooRealVar*)dataVars->find(_funcVal->GetName());

  // create iterators for the new sets
  TIterator *nextCatVar= catVars.MakeIterator();
  TIterator *nextRealVar= realVars.MakeIterator();

  // first generate enough events to get reasonable estimates for the integral and
  // maximum function value
  while(_totalEvents < _minTrials) addEvent(cache,nextCatVar,nextRealVar,funcVal);

  // increase our maximum estimate slightly to give a safety margin (and corresponding
  // loss of efficiency)
  _maxFuncVal*= 1.05;

  Int_t generatedEvts(0);
  while(generatedEvts < nEvents) {
    // Use any cached events first.
    if(acceptEvent(cache,funcVal,container)) {
      generatedEvts++;
    }
    else {
      // When we have used up the cache, start a new cache and add
      // some more events to it.      
      cache.Reset();
      _eventsUsed= 0;
      // Calculate how many more events to generate using our best estimate of our efficiency.
      // Always generate at least one more event so we don't get stuck.
      Int_t extra= 1 + (Int_t)((nEvents - generatedEvts)/eff());
      if(_verbose) cout << "generating " << extra << " events into reset cache" << endl;
      Double_t oldMax(_maxFuncVal);
      while(extra--) addEvent(cache,nextCatVar,nextRealVar,funcVal);
      if(_verbose && (_maxFuncVal > oldMax)) {
	cout << "RooAcceptReject::generateEvents: estimated function maximum increased from "
	     << oldMax << " to " << _maxFuncVal << endl;
      }
    }
  }

  // reattach our function clone to the cloned args
  _funcClone->recursiveRedirectServers(*_cloneSet,kTRUE);

  // cleanup
  delete nextRealVar;
  delete nextCatVar;
}

Bool_t RooAcceptReject::acceptEvent(const RooDataSet &cache, RooRealVar *funcVal, RooDataSet &container) {
  // Scan through events in the cache which have not been used yet,
  // looking for the first accepted one which is added to the specified
  // container. Return kTRUE if an accepted event is found, or otherwise
  // kFALSE. Update _eventsUsed.

  const RooArgSet *event(0);
  while(event= cache.get(++_eventsUsed)) {    
    // accept this cached event?
    Double_t r= RooRandom::uniform();
    if(r*_maxFuncVal > funcVal->getVal()) continue;
    // copy this event into the output container
    if(_verbose) cout << "accepted event (used " << _eventsUsed << " of "
		      << cache.GetEntries() << " so far)" << endl;
    container.add(*event);
    return kTRUE;
  }
  return kFALSE;
}

void RooAcceptReject::addEvent(RooDataSet &cache, TIterator *nextCatVar, TIterator *nextRealVar,
			       RooRealVar *funcVal) {
  // Add a trial event to the specified dataset and update our estimates
  // of the function maximum value and integral.

  // randomize each discrete argument
  nextCatVar->Reset();
  RooCategory *cat(0);
  while(cat= (RooCategory*)(*nextCatVar)()) cat->randomize();

  // randomize each real argument
  nextRealVar->Reset();
  RooRealVar *real(0);
  while(real= (RooRealVar*)(*nextRealVar)()) real->randomize();

  // calculate and store our function value at this new point
  Double_t val= _funcClone->getVal();
  funcVal->setVal(val);

  // update the estimated integral and maximum value
  if(val > _maxFuncVal) _maxFuncVal= val;
  _funcSum+= val;

  // fill a new entry in our cache dataset for this point
  cache.Fill();
  _totalEvents++;

  if(_verbose) cout << "=== [" << _totalEvents << "] " << val << " (I = "
		    << _funcSum/_totalEvents << ")" << endl;
}

const int RooAcceptReject::_maxSampleDim= 3,
  RooAcceptReject::_minTrialsArray[]= { 0,1000,100000,10000000 };
