/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooPlot.rdl,v 1.10 2001/05/14 22:54:21 verkerke Exp $
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

#include "RooFitCore/RooGenContext.hh"
#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooDataSet.hh"

#include "TRandom3.h"

ClassImp(RooGenContext)
  ;

static const char rcsid[] =
"$Id: RooPlot.cc,v 1.13 2001/05/14 22:56:53 david Exp $";

RooGenContext::RooGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype) :
  TNamed(model), _origVars(&vars), _prototype(prototype), _cloneSet(0), _pdfClone(0), _maxTrials(100)
{
  // Initialize a new context for generating events with the specified
  // variables, using the specified PDF model. A prototype dataset (if provided)
  // is not cloned and still belongs to the caller.

  // Clone the model and all nodes that it depends on so that this context
  // is independent of any existing objects.
  RooArgSet nodes(model,model.GetName());
  _cloneSet= nodes.snapshot(kTRUE);

  // Find the clone in the snapshot list
  _pdfClone = (RooAbsPdf*)_cloneSet->FindObject(model.GetName());

  // Analyze the list of variables to generate.
  _isValid= kTRUE;
  RooArgSet datasetVars("datasetVars"),generateVars("generateVars"),directVars("directVars");
  TIterator *iterator= vars.MakeIterator();
  TIterator *servers= _pdfClone->serverIterator();
  const RooAbsArg *arg(0);
  while(_isValid && (arg= (const RooAbsArg*)iterator->Next())) {
    // is the variable derived?
    if(arg->isDerived()) {
      cout << fName << "::" << ClassName() << ": cannot generate values for derived \""
	   << arg->GetName() << "\"" << endl;
      _isValid= kFALSE;
    }
    // does the model depend on this variable at all?
    if(!_cloneSet->contains(*arg)) {
      cout << fName << "::" << ClassName() << ":WARNING: model does not depend on \""
	   << arg->GetName() << "\"" << endl;
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

	// otherwise, this variable will have to be generated numerically
	_otherVars.add(*arg);
      }
    }
  }
  delete servers;
  delete iterator;

  // Analyze the prototype dataset, if one is specified
  if(_prototype) {
    iterator= _prototype->get()->MakeIterator();
    const RooAbsArg *proto(0);
    while(proto= (const RooAbsArg*)iterator->Next()) {
      // is this variable being generated or taken from the prototype?
      if(!_directVars.contains(*proto) && !_otherVars.contains(*proto)) {
	_protoVars.add(*proto);
      }
    }
    delete iterator;
  }

  // create a list of all variables that will appear in generated datasets
  _datasetVars.add(_directVars);
  _datasetVars.add(_otherVars);
  _datasetVars.add(_protoVars);
}

RooGenContext::~RooGenContext() {
  // Clean up the cloned objects used in this context.
  delete _cloneSet;
}

RooDataSet *RooGenContext::generate(Int_t nEvents) const {
  
  // create and initialize a new dataset
  RooDataSet *data= createDataset();

  // Reset the PDF's error counters
  _pdfClone->resetErrorCounters();

  // Calculate the expected number of events if requested
  if(nEvents <= 0) {
    nEvents= (Int_t)(_pdfClone->expectedEvents() + 0.5);
    if(nEvents <= 0) {
      cout << fName << "::" << ClassName()
	   << ":generate: cannot calculate expected number of events" << endl;
      return 0;
    }
  }

  // Loop over events to generate using our clone
  for(Int_t evt= 0; evt < nEvents; evt++) _pdfClone->generateEvent(*_origVars, _maxTrials);

  return data;
}

RooDataSet *RooGenContext::createDataset() const {
  // Create a new empty dataset using the specified variables and
  // attach our PDF clone to its variables.

  TString name(_pdfClone->GetName()),title(_pdfClone->GetTitle());
  name.Append("Data");
  title.Prepend("Generated From ");
  RooDataSet *data= new RooDataSet(name.Data(), title.Data(), *_origVars);

  // Attach our clone to the new data set
  _pdfClone->attachDataSet(data);

  return data;
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

TRandom &RooGenContext::randomGenerator() {
  // Return a pointer to a singleton random-number generator
  // implementation. Creates the object the first time it is called.

  static TRandom *_theGenerator= 0;
  if(0 == _theGenerator) _theGenerator= new TRandom3();
  return *_theGenerator;
}

Double_t RooGenContext::uniform() {
  // Return a number uniformly distributed from (0,1)

  return randomGenerator().Rndm();
}
