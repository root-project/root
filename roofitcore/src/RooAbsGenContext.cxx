/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsGenContext.cc,v 1.3 2001/10/13 23:02:17 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   11-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooAbsGenContext is the abstract base class for generator contexts.

#include "RooFitCore/RooAbsGenContext.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"

ClassImp(RooAbsGenContext)
;


RooAbsGenContext::RooAbsGenContext(const RooAbsPdf& model, const RooArgSet &vars,
				   const RooDataSet *prototype, Bool_t verbose) :
  TNamed(model), _prototype(prototype), _theEvent(0), _verbose(verbose), _isValid(kTRUE)
{
  // Constructor

  // Make a snapshot of the generated variables that we can overwrite.
  _theEvent= (RooArgSet*)vars.snapshot(kFALSE);

  // Analyze the prototype dataset, if one is specified
  _nextProtoIndex= 0;
  if(0 != _prototype) {
    TIterator *protoIterator= _prototype->get()->createIterator();
    const RooAbsArg *proto(0);
    while(proto= (const RooAbsArg*)protoIterator->Next()) {
      // is this variable being generated or taken from the prototype?
      if(!_theEvent->contains(*proto)) {
	_protoVars.add(*proto);
	_theEvent->addClone(*proto);
      }
    }
    delete protoIterator;
  }

  // Remember the default number of events to generate when no prototype dataset is provided.
  if (model.canBeExtended()) {
    _expectedEvents= (Int_t)(model.expectedEvents() + 0.5);
  } else {
    _expectedEvents= 0 ;
  }
}


RooAbsGenContext::~RooAbsGenContext()
{
  // Destructor

  if(0 != _theEvent) delete _theEvent;
}

RooDataSet *RooAbsGenContext::generate(Int_t nEvents) {
  // Generate the specified number of events with nEvents>0 and
  // and return a dataset containing the generated events. With nEvents<=0,
  // generate the number of events in the prototype dataset, if available,
  // or else the expected number of events, if non-zero. The returned
  // dataset belongs to the caller. Return zero in case of an error.
  // Generation of individual events is delegated to a virtual generateEvent()
  // method. A virtual initGenerator() method is also called just before the
  // first call to generateEvent().
  
  if(!isValid()) {
    cout << ClassName() << "::" << GetName() << ": context is not valid" << endl;
    return 0;
  }

  // Calculate the expected number of events if necessary
  if(nEvents <= 0) {
    if(_prototype) {
      nEvents= (Int_t)_prototype->numEntries();
    }
    else {
      nEvents= _expectedEvents;
    }
    if(nEvents <= 0) {
      cout << ClassName() << "::" << GetName()
	   << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
    else if(_verbose) {
      cout << ClassName() << "::" << GetName() << ":generate: will generate "
	   << nEvents << " events" << endl;
    }
  }

  // check that any prototype dataset still defines the variables we need
  // (this is necessary since we never make a private clone, for efficiency)
  if(_prototype) {
    const RooArgSet *vars= _prototype->get();
    TIterator *iterator= _protoVars.createIterator();
    const RooAbsArg *arg(0);
    Bool_t ok(kTRUE);
    while(arg= (const RooAbsArg*)iterator->Next()) {
      if(vars->contains(*arg)) continue;
      cout << ClassName() << "::" << GetName() << ":generate: prototype dataset is missing \""
	   << arg->GetName() << "\"" << endl;
      ok= kFALSE;
    }
    delete iterator;
    if(!ok) return 0;
  }

  // create a new dataset
  TString name(GetName()),title(GetTitle());
  name.Append("Data");
  title.Prepend("Generated From ");
  RooDataSet *data= new RooDataSet(name.Data(), title.Data(), *_theEvent);

  // Perform any subclass implementation-specific initialization
  initGenerator(*_theEvent);

  // Loop over the events to generate
  for(Int_t evt= 0; evt < nEvents; evt++) {

    // first, load values from the prototype dataset, if one was provided
    if(0 != _prototype) {
      if(_nextProtoIndex >= _prototype->numEntries()) _nextProtoIndex= 0;
      const RooArgSet *subEvent= _prototype->get(_nextProtoIndex);
      _nextProtoIndex++;
      if(0 != subEvent) {
	*_theEvent= *subEvent;
      }
      else {
	cout << ClassName() << "::" << GetName() << ":generate: cannot load event "
	     << _nextProtoIndex << " from prototype dataset" << endl;
	return 0;
      }
    }

    // delegate the generation of the rest of this event to our subclass implementation
    generateEvent(*_theEvent, nEvents - evt);

    data->add(*_theEvent);
  }
  
  return data;
}

void RooAbsGenContext::initGenerator(const RooArgSet &theEvent) {
  // The base class provides a do-nothing default implementation.
}

void RooAbsGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  oneLinePrint(os,*this);
  if(opt >= Standard) {
    PrintOption less= lessVerbose(opt);
    TString deeper(indent);
    indent.Append("  ");
    os << indent << "  Generator of ";
    _theEvent->printToStream(os,less,deeper);
    os << indent << "  Prototype variables are ";
    _protoVars.printToStream(os,less,deeper);
  }
}
