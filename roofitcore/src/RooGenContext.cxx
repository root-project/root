/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.cc,v 1.16 2001/09/28 21:59:28 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-May-2000 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// A class description belongs here...

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooGenContext.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooAcceptReject.hh"

#include "TString.h"
#include "TIterator.h"

ClassImp(RooGenContext)
  ;

static const char rcsid[] =
"$Id: RooGenContext.cc,v 1.16 2001/09/28 21:59:28 verkerke Exp $";

RooGenContext::RooGenContext(const RooAbsPdf &model, const RooArgSet &vars,
			     const RooDataSet *prototype, Bool_t verbose) :
  TNamed(model), _origVars(&vars), _prototype(prototype), _cloneSet(0), _pdfClone(0),
  _acceptRejectFunc(0), _generator(0), _verbose(verbose)
{
  // Initialize a new context for generating events with the specified
  // variables, using the specified PDF model. A prototype dataset (if provided)
  // is not cloned and still belongs to the caller. The contents and shape
  // of this dataset can be changed between calls to generate() as long as the
  // expected columns to be copied to the generated dataset are present.

  // Clone the model and all nodes that it depends on so that this context
  // is independent of any existing objects.
  RooArgSet nodes(model,model.GetName());
  _cloneSet= (RooArgSet*) nodes.snapshot(kTRUE);

  // Find the clone in the snapshot list
  _pdfClone = (RooAbsPdf*)_cloneSet->find(model.GetName());


  // Analyze the list of variables to generate...
  _isValid= kTRUE;
  TIterator *iterator= vars.createIterator();
  TIterator *servers= _pdfClone->serverIterator();
  const RooAbsArg *tmp(0),*arg(0);
  while(_isValid && (tmp= (const RooAbsArg*)iterator->Next())) {
    // is this argument derived?
    if(tmp->isDerived()) {
      cout << fName << "::" << ClassName() << ": cannot generate values for derived \""
	   << tmp->GetName() << "\"" << endl;
      _isValid= kFALSE;
      continue;
    }
    // lookup this argument in the cloned set of PDF dependents
    arg= (const RooAbsArg*)_cloneSet->find(tmp->GetName());
    if(0 == arg) {
      cout << fName << "::" << ClassName() << ":WARNING: model does not depend on \""
	   << tmp->GetName() << "\" which will have uniform distribution" << endl;
      _uniformVars.add(*tmp);
    }
    else {
      // does the model depend on this variable directly, ie, like "x" in
      // f(x) or f(x,g(x,y)) or even f(x,x) ?
      RooAbsArg *direct= _pdfClone->findServer(arg->GetName());
      if(direct) {
	// is this the only way that the model depends on this variable?
	servers->Reset();
	const RooAbsArg *server(0);
	while(direct && (server= (const RooAbsArg*)servers->Next())) {
	  if(server == direct) continue;
	  if(server->dependsOn(*arg)) direct= 0;
	}
	if(direct) {
	  _directVars.add(*arg);
	}
	else {
	  _otherVars.add(*arg);
	}
      }
      else {
	// does the model depend indirectly on this variable through an lvalue chain?
	
	// otherwise, this variable will have to be generated with accept/reject
	_otherVars.add(*arg);
      }
    }
  }
  delete servers;
  delete iterator;
  if(!_isValid) {
    cout << fName << "::" << ClassName() << ": constructor failed with errors" << endl;
    return;
  }

  // Analyze the prototype dataset, if one is specified
  if(_prototype) {
    iterator= _prototype->get()->createIterator();
    const RooAbsArg *proto(0);
    while(proto= (const RooAbsArg*)iterator->Next()) {
      // is this variable being generated or taken from the prototype?
      if(!_directVars.contains(*proto) && !_otherVars.contains(*proto) && !_uniformVars.contains(*proto)) {
	_protoVars.add(*proto);
      }
    }
    delete iterator;
  }

  // Can the model generate any of the direct variables itself?
  RooArgSet generatedVars;
  _code= _pdfClone->getGenerator(_directVars,generatedVars);
  // Move variables which cannot be generated into the list to be generated with accept/reject
  _directVars.remove(generatedVars);
  _otherVars.add(_directVars);
  _directVars.removeAll();
  _directVars.add(generatedVars);

  // create a list of all variables that will appear in generated datasets
  _datasetVars.add(_directVars);
  _datasetVars.add(_otherVars);
  _datasetVars.add(_uniformVars);
  _datasetVars.add(_protoVars);

  // initialize the accept-reject generator
  RooArgSet *depList= _pdfClone->getDependents(&_datasetVars);
  depList->remove(_otherVars);
  TString nname(_pdfClone->GetName()) ;
  nname.Append("Reduced") ;
  TString ntitle(_pdfClone->GetTitle()) ;
  ntitle.Append(" (Accept/Reject)") ;
  _acceptRejectFunc= new RooRealIntegral(nname,ntitle,*_pdfClone,*depList,&vars);
  delete depList;
  _otherVars.add(_uniformVars);
  _generator= new RooAcceptReject(*_acceptRejectFunc,_otherVars,_verbose);
}

RooGenContext::~RooGenContext() {
  // Clean up the cloned objects used in this context.
  delete _cloneSet;
  // Clean up our accept/reject generator
  delete _generator;
  delete _acceptRejectFunc;
}

RooDataSet *RooGenContext::generate(Int_t nEvents) const {
  // Generate the specified number of events with nEvents>0 and
  // and return a dataset containing the generated events. With nEvents<=0,
  // generate the number of events in the prototype dataset, if available,
  // or else the expected number of events, if non-zero. The returned
  // dataset belongs to the caller. Return zero in case of an error.
  
  if(!_isValid) {
    cout << fName << "::" << ClassName() << ": context is not valid" << endl;
    return 0;
  }

  // Calculate the expected number of events if necessary
  if(nEvents <= 0) {
    if(_prototype) {
      nEvents= (Int_t)_prototype->numEntries();
    }
    else {
      nEvents= (Int_t)(_pdfClone->expectedEvents() + 0.5);
    }
    if(nEvents <= 0) {
      cout << fName << "::" << ClassName()
	   << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
  }

  // check that the dataset still defines the variables we need
  // (this is necessary since we never make a private clone for efficiency)
  if(_prototype) {
    const RooArgSet *vars= _prototype->get();
    TIterator *iterator= _protoVars.createIterator();
    const RooAbsArg *arg(0);
    Bool_t ok(kTRUE);
    while(arg= (const RooAbsArg*)iterator->Next()) {
      if(vars->contains(*arg)) continue;
      cout << fName << "::" << ClassName() << ":generate: prototype dataset is missing \""
	   << arg->GetName() << "\"" << endl;
      ok= kFALSE;
    }
    delete iterator;
    if(!ok) return 0;
  }

  // create a new dataset
  TString name(_pdfClone->GetName()),title(_pdfClone->GetTitle());
  name.Append("Data");
  title.Prepend("Generated From ");
  RooDataSet *data= new RooDataSet(name.Data(), title.Data(), _datasetVars);

  // preload the dataset with values from our accept-reject generator
  _generator->generateEvents(nEvents,*data);

  // WVE should this be here?
  return data;

//   // Attach the model to the new data set
//   _pdfClone->attachDataSet(*data);

//   // Reset the PDF's error counters
//   _pdfClone->resetErrorCounters();

//   // Loop over the events to generate
//   for(Int_t evt= 0; evt < nEvents; evt++) {
//     // load values from the prototype dataset if requested
//     if(_prototype) {
//       //...
//     }
//     // generate values of the variables that the model cannot generate itself
//     //acceptReject();
//     // use the model's generator
//     _pdfClone->generateEvent(_code);
//     // add this event to the dataset
//     data->fill();
//   }

//   return data;
}

void RooGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  oneLinePrint(os,*this);
  if(opt >= Standard) {
    PrintOption less= lessVerbose(opt);
    os << "Generator of ";
    _datasetVars.printToStream(os,less,indent);
    os << indent << "Using PDF ";
    _pdfClone->printToStream(os,less,indent);
    if(opt >= Verbose) {
      os << indent << "PDF depends on ";
      _cloneSet->printToStream(os,less,indent);
      os << indent << "Use PDF generator for ";
      _directVars.printToStream(os,less,indent);
      os << indent << "Use accept/reject for ";
      _otherVars.printToStream(os,less,indent);
      os << indent << "Use prototype data for ";
      _protoVars.printToStream(os,less,indent);
    }
  }
}
