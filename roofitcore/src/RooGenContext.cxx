/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooGenContext.cc,v 1.27 2002/03/22 22:43:55 verkerke Exp $
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
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooDataHist.hh"
#include "RooFitCore/RooErrorHandler.hh"

#include "TString.h"
#include "TIterator.h"

ClassImp(RooGenContext)
  ;

static const char rcsid[] =
"$Id: RooGenContext.cc,v 1.27 2002/03/22 22:43:55 verkerke Exp $";

RooGenContext::RooGenContext(const RooAbsPdf &model, const RooArgSet &vars,
			     const RooDataSet *prototype, Bool_t verbose,
			     const RooArgSet* forceDirect) :  
  RooAbsGenContext(model,vars,prototype,verbose),
  _cloneSet(0), _pdfClone(0), _acceptRejectFunc(0), _generator(0),
  _maxVar(0), _uniIter(0) 
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
  if (!_cloneSet) {
    cout << "RooGenContext::RooGenContext(" << GetName() << ") Couldn't deep-clone PDF, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }

  // Find the clone in the snapshot list
  _pdfClone = (RooAbsPdf*)_cloneSet->find(model.GetName());

  // Analyze the list of variables to generate...
  _isValid= kTRUE;
  TIterator *iterator= vars.createIterator();
  TIterator *servers= _pdfClone->serverIterator();
  const RooAbsArg *tmp = 0;
  const RooAbsArg *arg = 0;
  while(_isValid && (tmp= (const RooAbsArg*)iterator->Next())) {
    // is this argument derived?
    if(tmp->isDerived()) {
      cout << ClassName() << "::" << GetName() << ": cannot generate values for derived \""
	   << tmp->GetName() << "\"" << endl;
      _isValid= kFALSE;
      continue;
    }
    // lookup this argument in the cloned set of PDF dependents
    arg= (const RooAbsArg*)_cloneSet->find(tmp->GetName());
    if(0 == arg) {
      cout << ClassName() << "::" << GetName() << ":WARNING: model does not depend on \""
	   << tmp->GetName() << "\" which will have uniform distribution" << endl;
      _uniformVars.add(*tmp);
    }
    else {
      // does the model depend on this variable directly, ie, like "x" in
      // f(x) or f(x,g(x,y)) or even f(x,x) ?
      RooAbsArg *direct= _pdfClone->findServer(arg->GetName());
      if(direct) {

	if (forceDirect==0 || !forceDirect->find(direct->GetName())) {
	  if (!_pdfClone->isDirectGenSafe(*arg)) direct=0 ;
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
  if(!isValid()) {
    cout << ClassName() << "::" << GetName() << ": constructor failed with errors" << endl;
    return;
  }

  // Can the model generate any of the direct variables itself?
  RooArgSet generatedVars;
  _code= _pdfClone->getGenerator(_directVars,generatedVars);

  // Move variables which cannot be generated into the list to be generated with accept/reject
  _directVars.remove(generatedVars);
  _otherVars.add(_directVars);

  // Update _directVars to only include variables that will actually be directly generated
  _directVars.removeAll();
  _directVars.add(generatedVars);

  // initialize the accept-reject generator
  RooArgSet *depList= _pdfClone->getDependents(_theEvent);
  depList->remove(_otherVars);

  TString nname(_pdfClone->GetName()) ;
  nname.Append("Reduced") ;
  TString ntitle(_pdfClone->GetTitle()) ;
  ntitle.Append(" (Accept/Reject)") ;

  if (_protoVars.getSize()==0) {

    // No prototype variables
    _acceptRejectFunc= new RooRealIntegral(nname,ntitle,*_pdfClone,*depList,&vars);

  } else {

    // Generation with prototype variable
    depList->remove(_protoVars,kTRUE,kTRUE) ;
    _acceptRejectFunc= new RooRealIntegral(nname,ntitle,*_pdfClone,*depList,&vars);

    // First find maximum in other+proto space
    RooArgSet otherAndProto(_otherVars) ;
    otherAndProto.add(_protoVars) ;

    RooAcceptReject maxFinder(*_acceptRejectFunc,otherAndProto,0,_verbose) ;
    Double_t max = maxFinder.getFuncMax() ;
    _maxVar = new RooRealVar("funcMax","function maximum",max) ;
  }

  _generator= new RooAcceptReject(*_acceptRejectFunc,_otherVars,_maxVar,_verbose);

  delete depList;
  _otherVars.add(_uniformVars);
}

RooGenContext::~RooGenContext() 
{
  // Destructor.

  // Clean up the cloned objects used in this context.
  delete _cloneSet;
  // Clean up our accept/reject generator
  delete _generator;
  delete _acceptRejectFunc;
  if (_maxVar) delete _maxVar ;
  if (_uniIter) delete _uniIter ;
}

void RooGenContext::initGenerator(const RooArgSet &theEvent) {

  // Attach the cloned model to the event buffer we will be filling.
  _pdfClone->recursiveRedirectServers(theEvent,kFALSE);

  // Attach the RooAcceptReject generator the event buffer
  _generator->attachParameters(theEvent) ;

  // Reset the cloned model's error counters.
  _pdfClone->resetErrorCounters();

  // Initialize the PDFs internal generator
  if (_directVars.getSize()>0) {
    _pdfClone->initGenerator(_code) ;
  }

  // Create iterator for uniform vars (if any)
  if (_uniformVars.getSize()>0) {
    _uniIter = _uniformVars.createIterator() ;
  }
}

void RooGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining) {
  // Generate variables for a new event.

  if(_otherVars.getSize() > 0) {
    // call the accept-reject generator to generate its variables
    const RooArgSet *subEvent= _generator->generateEvent(remaining);
    if(0 == subEvent) {
      cout << ClassName() << "::" << GetName() << ":generate: accept/reject generator failed." << endl;
      return;
    }
    theEvent= *subEvent;
  }

  // Use the model's optimized generator, if one is available.
  // The generator writes directly into our local 'event' since we attached it above.
  if(_directVars.getSize() > 0) {
    _pdfClone->generateEvent(_code);
  }

  // Generate uniform variables (non-dependents)  
  if (_uniIter) {
    _uniIter->Reset() ;
    RooAbsArg* uniVar ;
    while(uniVar=(RooAbsArg*)_uniIter->Next()) {
      RooAbsLValue* arglv = dynamic_cast<RooAbsLValue*>(uniVar) ;
      if (!arglv) {
	cout << "RooGenContext::generateEvent(" << GetName() << ") ERROR: uniform variable " 
	     << uniVar->GetName() << " is not an lvalue" << endl ;
	RooErrorHandler::softAbort() ;
      }
      arglv->randomize() ;
    }
    theEvent = _uniformVars ;
  }
}

void RooGenContext::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  RooAbsGenContext::printToStream(os,opt,indent);
  if(opt >= Standard) {
    PrintOption less= lessVerbose(opt);
    TString deeper(indent);
    indent.Append("  ");
    os << indent << "Using PDF ";
    _pdfClone->printToStream(os,less,deeper);
    if(opt >= Verbose) {
      os << indent << "Use PDF generator for ";
      _directVars.printToStream(os,less,deeper);
      os << indent << "Use accept/reject for ";
      _otherVars.printToStream(os,less,deeper);
    }
  }
}
