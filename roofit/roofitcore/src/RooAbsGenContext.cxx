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
\file RooAbsGenContext.cxx
\class RooAbsGenContext
\ingroup Roofitcore

RooAbsGenContext is the abstract base class for generator contexts of
RooAbsPdf objects. A generator context is an object that controls
the generation of events from a given p.d.f in one or more sessions.
This class defines the common interface for all such contexts and organizes
storage of common components, such as the observables definition, the
prototype data etc..
**/

#include "TClass.h"

#include "RooAbsGenContext.h"
#include "RooRandom.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooMsgService.h"
#include "RooGlobalFunc.h"

#include "Riostream.h"


using namespace std;

ClassImp(RooAbsGenContext);
;


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsGenContext::RooAbsGenContext(const RooAbsPdf& model, const RooArgSet &vars,
               const RooDataSet *prototype, const RooArgSet* auxProto, bool verbose) :
  TNamed(model),
  _prototype(prototype),
  _isValid(true),
  _verbose(verbose),
  _protoOrder(0),
  _genData(0)
{
  // Check PDF dependents
  if (model.recursiveCheckObservables(&vars)) {
    coutE(Generation) << "RooAbsGenContext::ctor: Error in PDF dependents" << endl ;
    _isValid = false ;
    return ;
  }

  // Make a snapshot of the generated variables that we can overwrite.
  vars.snapshot(_theEvent, false);

  // Analyze the prototype dataset, if one is specified
  _nextProtoIndex= 0;
  if(0 != _prototype) {
    TIterator *protoIterator= _prototype->get()->createIterator();
    const RooAbsArg *proto = 0;
    while((proto= (const RooAbsArg*)protoIterator->Next())) {
      // is this variable being generated or taken from the prototype?
      if(!_theEvent.contains(*proto)) {
   _protoVars.add(*proto);
   _theEvent.addClone(*proto);
      }
    }
    delete protoIterator;
  }

  // Add auxiliary protovars to _protoVars, if provided
  if (auxProto) {
    _protoVars.add(*auxProto) ;
    _theEvent.addClone(*auxProto);
  }

  // Remember the default number of events to generate when no prototype dataset is provided.
  _extendMode = model.extendMode() ;
  if (model.canBeExtended()) {
    _expectedEvents= (Int_t)(model.expectedEvents(&_theEvent) + 0.5);
  } else {
    _expectedEvents= 0 ;
  }

  // Save normalization range
  if (model.normRange()) {
    _normRange = model.normRange() ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsGenContext::~RooAbsGenContext()
{
  if (_protoOrder) delete[] _protoOrder ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface to attach given parameters to object in this context

void RooAbsGenContext::attach(const RooArgSet& /*params*/)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Create an empty dataset to hold the events that will be generated

RooDataSet* RooAbsGenContext::createDataSet(const char* name, const char* title, const RooArgSet& obs)
{
  RooDataSet* ret = new RooDataSet(name, title, obs);
  ret->setDirtyProp(false) ;
  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Generate the specified number of events with nEvents>0 and
/// and return a dataset containing the generated events. With nEvents<=0,
/// generate the number of events in the prototype dataset, if available,
/// or else the expected number of events, if non-zero.
/// If extendedMode = true generate according to a Poisson(nEvents)
/// The returned dataset belongs to the caller. Return zero in case of an error.
/// Generation of individual events is delegated to a virtual generateEvent()
/// method. A virtual initGenerator() method is also called just before the
/// first call to generateEvent().

RooDataSet *RooAbsGenContext::generate(double nEvents, bool skipInit, bool extendedMode)
{
  if(!isValid()) {
    coutE(Generation) << ClassName() << "::" << GetName() << ": context is not valid" << endl;
    return 0;
  }

  // Calculate the expected number of events if necessary
  if(nEvents <= 0) {
    if(_prototype) {
      nEvents= (Int_t)_prototype->numEntries();
    }
    else {
      if (_extendMode == RooAbsPdf::CanNotBeExtended) {
   coutE(Generation) << ClassName() << "::" << GetName()
        << ":generate: PDF not extendable: cannot calculate expected number of events" << endl;
   return 0;
      }
      nEvents= _expectedEvents;
    }
    if(nEvents <= 0) {
      coutE(Generation) << ClassName() << "::" << GetName()
         << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
    coutI(Generation) << ClassName() << "::" << GetName() << ":generate: will generate "
            << nEvents << " events" << endl;

  }

  if (extendedMode) {
     double nExpEvents = nEvents;
     nEvents = RooRandom::randomGenerator()->Poisson(nEvents) ;
     cxcoutI(Generation) << " Extended mode active, number of events generated (" << nEvents << ") is Poisson fluctuation on "
                         << GetName() << "::expectedEvents() = " << nExpEvents << endl ;
  }

  // check that any prototype dataset still defines the variables we need
  // (this is necessary since we never make a private clone, for efficiency)
  if(_prototype) {
    const RooArgSet *vars= _prototype->get();
    TIterator *iterator= _protoVars.createIterator();
    const RooAbsArg *arg = 0;
    bool ok(true);
    while((arg= (const RooAbsArg*)iterator->Next())) {
      if(vars->contains(*arg)) continue;
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":generate: prototype dataset is missing \""
             << arg->GetName() << "\"" << endl;

      // WVE disable this for the moment
      // ok= false;
    }
    delete iterator;
    // coverity[DEADCODE]
    if(!ok) return 0;
  }

  if (_verbose) Print("v") ;

  // create a new dataset
  TString name(GetName()),title(GetTitle());
  name.Append("Data");
  title.Prepend("Generated From ");

  // WVE need specialization here for simultaneous pdfs
  _genData = createDataSet(name.Data(), title.Data(), _theEvent);

  // Perform any subclass implementation-specific initialization
  // Can be skipped if this is a rerun with an identical configuration
  if (!skipInit) {
    initGenerator(_theEvent);
  }

  // Loop over the events to generate
  Int_t evt(0) ;
  while(_genData->numEntries()<nEvents) {

    // first, load values from the prototype dataset, if one was provided
    if(0 != _prototype) {
      if(_nextProtoIndex >= _prototype->numEntries()) _nextProtoIndex= 0;

      Int_t actualProtoIdx = _protoOrder ? _protoOrder[_nextProtoIndex] : _nextProtoIndex ;

      const RooArgSet *subEvent= _prototype->get(actualProtoIdx);
      _nextProtoIndex++;
      if(0 != subEvent) {
        _theEvent.assign(*subEvent);
      }
      else {
   coutE(Generation) << ClassName() << "::" << GetName() << ":generate: cannot load event "
           << actualProtoIdx << " from prototype dataset" << endl;
   return 0;
      }
    }

    // delegate the generation of the rest of this event to our subclass implementation
    generateEvent(_theEvent, (Int_t)(nEvents - _genData->numEntries()));


    // WVE add check that event is in normRange
    if (_normRange.Length()>0 && !_theEvent.isInRange(_normRange.Data())) {
      continue ;
    }

    _genData->addFast(_theEvent);
    evt++ ;
  }

  RooDataSet* output = _genData ;
  _genData = 0 ;
  output->setDirtyProp(true) ;

  return output;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function to initialize context for generation for given
/// set of observables

void RooAbsGenContext::initGenerator(const RooArgSet&)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Print name of context

void RooAbsGenContext::printName(ostream& os) const
{
  os << GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print title of context

void RooAbsGenContext::printTitle(ostream& os) const
{
  os << GetTitle() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print class name of context

void RooAbsGenContext::printClassName(ostream& os) const
{
  os << ClassName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print arguments of context, i.e. the observables being generated in this context

void RooAbsGenContext::printArgs(ostream& os) const
{
  os << "[ " ;
  TIterator* iter = _theEvent.createIterator() ;
  RooAbsArg* arg ;
  bool first(true) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (first) {
      first=false ;
    } else {
      os << "," ;
    }
    os << arg->GetName() ;
  }
  os << "]" ;
  delete iter ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface for multi-line printing

void RooAbsGenContext::printMultiline(ostream &/*os*/, Int_t /*contents*/, bool /*verbose*/, TString /*indent*/) const
{
}




////////////////////////////////////////////////////////////////////////////////
/// Set the traversal order of prototype data to that in the lookup tables
/// passed as argument. The LUT must be an array of integers with the same
/// size as the number of entries in the prototype dataset and must contain
/// integer values in the range [0,Nevt-1]

void RooAbsGenContext::setProtoDataOrder(Int_t* lut)
{
  // Delete any previous lookup table
  if (_protoOrder) {
    delete[] _protoOrder ;
    _protoOrder = 0 ;
  }

  // Copy new lookup table if provided and needed
  if (lut && _prototype) {
    Int_t n = _prototype->numEntries() ;
    _protoOrder = new Int_t[n] ;
    Int_t i ;
    for (i=0 ; i<n ; i++) {
      _protoOrder[i] = lut[i] ;
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Rescale existing output buffer with given ratio

void RooAbsGenContext::resampleData(double& ratio)
{

  Int_t nOrig = _genData->numEntries() ;
  Int_t nTarg = Int_t(nOrig*ratio+0.5) ;
  RooDataSet* trimmedData = (RooDataSet*) _genData->reduce(RooFit::EventRange(0,nTarg)) ;

  cxcoutD(Generation) << "RooGenContext::resampleData*( existing production trimmed from " << nOrig << " to " << trimmedData->numEntries() << " events" << endl ;

  delete _genData ;
  _genData = trimmedData ;

  if (_prototype) {
    // Push back proto index by trimmed amount to force recycling of the
    // proto entries that were trimmed away
    _nextProtoIndex -= (nOrig-nTarg) ;
    while (_nextProtoIndex<0) {
      _nextProtoIndex += _prototype->numEntries() ;
    }
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Define default contents when printing

Int_t RooAbsGenContext::defaultPrintContents(Option_t* /*opt*/) const
{
  return kName|kClassName|kValue ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define default print style

RooPrintable::StyleOption RooAbsGenContext::defaultPrintStyle(Option_t* opt) const
{
  if (opt && TString(opt).Contains("v")) {
    return kVerbose ;
  }
  return kStandard ;
}
