/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.cc,v 1.1 2001/05/18 00:59:19 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A class description belongs here...

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooAcceptReject.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooDataSet.hh"

#include "TString.h"
#include "TIterator.h"

ClassImp(RooAcceptReject)
  ;

static const char rcsid[] =
"$Id: RooGenContext.cc,v 1.1 2001/05/18 00:59:19 david Exp $";

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
  // sets of categories and reals. Calculate the area of the generating
  // space and check that it is finite.
  _area= 1;
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
    }
    else {
      // clone any variables we generate that we haven't cloned already
      arg= (RooAbsArg*)arg->Clone();
      _cloneSet->add(*arg);
    }
    // is this argument a category or a real?
    const RooCategory *catVar= dynamic_cast<const RooCategory*>(arg);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(arg);
    if(0 != catVar) {
      _catVars.add(*catVar);
      _area*= catVar->numTypes();
    }
    else if(0 != realVar) {
      if(realVar->hasFitMin() && realVar->hasFitMax()) {
	_realVars.add(*realVar);
	_area*= realVar->getFitMax() - realVar->getFitMin();
	if(found) _sampleDim++;
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
  _minTrials= 1;
  for(Int_t dim= 0; dim < _sampleDim && dim < 3; dim++) _minTrials*= 100;
  if(_sampleDim > 3) {
    cout << fName << "::" << ClassName() << ": WARNING: generating " << _sampleDim
	 << " variables with accept-reject may not be accurate" << endl;
  }
  // print a verbose summary of our configuration, if requested
  if(_verbose) {
    cout << fName << "::" << ClassName() << ":" << endl
	 << "  Initializing accept-reject generator for ";
    _funcClone->Print();
    cout << "  Accept-Reject area is " << _area << endl;
    cout << "  Sampling dimension is " << _sampleDim << endl;
    cout << "  Min sampling trials is " << _minTrials << endl;
    cout << "  Will generate category vars ";
    TString indent("  ");
    _catVars.printToStream(cout,Standard,indent);
    cout << "  Will generate real vars ";
    _realVars.printToStream(cout,Standard,indent);
  }

  // create a fundamental type for storing function values
  _funcVal= new RooRealVar(TString(_funcClone->GetName()).Append("Val"),
			   TString(_funcClone->GetTitle()).Append(" Function Value"),0);

  // initialize our statistics
  _maxFuncVal= 0;
  _funcNorm= 0;
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

  // attach our function clone to this dataset
  _funcClone->recursiveRedirectServers(*cache.get(),kFALSE);

  // create new sets of category and real args that refer to the new dataset
  const RooArgSet *dataVars= cache.get();
  RooArgSet catVars(_catVars),realVars(_realVars);
  catVars.replace(*dataVars);
  realVars.replace(*dataVars);

  // create iterators for the new sets
  TIterator *nextCatVar= catVars.MakeIterator();
  TIterator *nextRealVar= realVars.MakeIterator();

  // first generate enough events to get reasonable estimates for the integral and
  // maximum function value

  _minTrials= 1;

  while(_totalEvents < _minTrials) addEvent(cache,nextCatVar,nextRealVar);

  // reattach our function clone to the cloned args
  _funcClone->recursiveRedirectServers(*_cloneSet,kTRUE);

  cache.Print("V");

  // cleanup
  delete nextRealVar;
  delete nextCatVar;
}

void RooAcceptReject::addEvent(RooDataSet &cache, TIterator *nextCatVar, TIterator *nextRealVar) {
  // Add a trial event to the specified dataset and update our estimates
  // of the function maximum value and integral.

  // randomize each discrete argument
  nextCatVar->Reset();
  RooCategory *cat(0);
  while(cat= (RooCategory*)(*nextCatVar)()) cat->randomize();

  // randomize each real argument
  nextRealVar->Reset();
  RooRealVar *real(0);
  RooAbsArg::verboseDirty(kTRUE);
  while(real= (RooRealVar*)(*nextRealVar)()) { real->randomize(); real->Print("V"); }

  // calculate and store our function value
  Double_t val= _funcClone->getVal();
  _funcClone->Print("V");
  _funcVal->setVal(val);

  // update the estimated integral and maximum value
  if(val > _maxFuncVal) _maxFuncVal= val;

  cache.Fill();
  _totalEvents++;

  cache.get()->Print("V");
}
