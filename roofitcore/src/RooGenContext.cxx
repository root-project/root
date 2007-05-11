/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGenContext.cc,v 1.48 2006/07/04 15:07:57 wverkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// A class description belongs here...


#include "RooFit.h"

#include "RooGenContext.h"
#include "RooGenContext.h"
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooRealIntegral.h"
#include "RooAcceptReject.h"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooErrorHandler.h"

#include "TString.h"
#include "TIterator.h"

ClassImp(RooGenContext)
  ;


RooGenContext::RooGenContext(const RooAbsPdf &model, const RooArgSet &vars,
			     const RooDataSet *prototype, const RooArgSet* auxProto,
			     Bool_t verbose, const RooArgSet* forceDirect) :  
  RooAbsGenContext(model,vars,prototype,auxProto,verbose),
  _cloneSet(0), _pdfClone(0), _acceptRejectFunc(0), _generator(0),
  _maxVar(0), _uniIter(0), _updateFMaxPerEvent(0) 
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

  // Optionally fix RooAddPdf normalizations
  if (prototype&&_pdfClone->dependsOn(*prototype->get())) {
    RooArgSet fullNormSet(vars) ;
    fullNormSet.add(*prototype->get()) ;
    _pdfClone->fixAddCoefNormalization(fullNormSet) ;
  }
      
  // Analyze the list of variables to generate...
  _isValid= kTRUE;
  TIterator *iterator= vars.createIterator();
  TIterator *servers= _pdfClone->serverIterator();
  const RooAbsArg *tmp = 0;
  const RooAbsArg *arg = 0;
  while((_isValid && (tmp= (const RooAbsArg*)iterator->Next()))) {
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
      const RooAbsArg *direct= arg ;
      if (forceDirect==0 || !forceDirect->find(direct->GetName())) {
	if (!_pdfClone->isDirectGenSafe(*arg)) {
	  //cout << "isDirectGenSafe on pdf " << _pdfClone->GetName() << " fails for arg " << arg->GetName() << endl ;
	  direct=0 ;
	}
      }
      
      // does the model depend indirectly on this variable through an lvalue chain?	
      // otherwise, this variable will have to be generated with accept/reject
      if(direct) { 
	_directVars.add(*arg);
      } else {
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

  // If PDF depends on prototype data, direct generator cannot use static initialization
  // in initGenerator()
  Bool_t staticInitOK = !_pdfClone->dependsOn(_protoVars) ;
  
  // Can the model generate any of the direct variables itself?
  RooArgSet generatedVars;
  _code= _pdfClone->getGenerator(_directVars,generatedVars,staticInitOK);

  // Move variables which cannot be generated into the list to be generated with accept/reject
  _directVars.remove(generatedVars);
  _otherVars.add(_directVars);

  // Update _directVars to only include variables that will actually be directly generated
  _directVars.removeAll();
  _directVars.add(generatedVars);

  // initialize the accept-reject generator
  RooArgSet *depList= _pdfClone->getObservables(_theEvent);
  depList->remove(_otherVars);

  TString nname(_pdfClone->GetName()) ;
  nname.Append("Reduced") ;
  TString ntitle(_pdfClone->GetTitle()) ;
  ntitle.Append(" (Accept/Reject)") ;

  if (_protoVars.getSize()==0) {

    // No prototype variables
    
    if(depList->getSize()==0) {
      // All variable are generated with accept-reject
      
      // Check if PDF supports maximum finding
      Int_t maxFindCode = _pdfClone->getMaxVal(_otherVars) ;
      if (maxFindCode != 0) {
	if (verbose) {
	  cout << "RooGenContext::ctor(" << model.GetName() 
	       << ") PDF supports maximum finding, initial sampling phase skipped" << endl ;
	}
	Double_t maxVal = _pdfClone->maxVal(maxFindCode) / _pdfClone->getNorm(_theEvent) ;
	_maxVar = new RooRealVar("funcMax","function maximum",maxVal) ;
      }
    } 

    _pdfClone->getVal(&vars) ; // WVE debug
    _acceptRejectFunc= new RooRealIntegral(nname,ntitle,*_pdfClone,*depList,&vars);

  } else {

    // Generation with prototype variable
    depList->remove(_protoVars,kTRUE,kTRUE) ;
    _acceptRejectFunc= (RooRealIntegral*) _pdfClone->createIntegral(*depList,vars) ;
    // _acceptRejectFunc=new RooRealIntegral(nname,ntitle,*_pdfClone,*depList,&vars);
//     cout << "RooGenContext:: creating a/c func from pdfClone with integrands " << *depList << " and norm vars = " << vars << endl ;

    if (_directVars.getSize()==0)  {

      // Check if PDF supports maximum finding
      Int_t maxFindCode = _pdfClone->getMaxVal(_otherVars) ;
      if (maxFindCode != 0) {

	// Special case: PDF supports max-finding in otherVars, no need to scan other+proto space for maximum
	if (verbose) {
	  cout << "RooGenContext::ctor(" << model.GetName() 
	       << ") PDF supports maximum finding, initial sampling phase skipped" << endl ;
	}
	_maxVar = new RooRealVar("funcMax","function maximum",1) ;
	_updateFMaxPerEvent = maxFindCode ;
      }
    }


    
    if (!_maxVar) {
//       cout << "RooGenContext::ctor _maxVar is null, _otherVars = " ; _otherVars.Print("1") ;

      // Regular case: First find maximum in other+proto space
      RooArgSet otherAndProto(_otherVars) ;
      
      RooArgSet* protoDeps = model.getObservables(_protoVars) ;
      otherAndProto.add(*protoDeps) ;
      delete protoDeps ;
      
      if (_otherVars.getSize()>0) {      
	// Calculate maximum in other+proto space if there are any accept/reject generated observables
// 	cout << "RooGenContext::ctor maxFinder config: scan in otherAndProto = " ; otherAndProto.Print("1") ;
	RooAcceptReject maxFinder(*_acceptRejectFunc,otherAndProto,0,_verbose) ;
	Double_t max = maxFinder.getFuncMax() ;
	_maxVar = new RooRealVar("funcMax","function maximum",max) ;
// 	cout << "RooGenContext::ctor _maxVar value set to " << max << endl ;
      }
    }
      
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
  _acceptRejectFunc->recursiveRedirectServers(theEvent,kFALSE) ; // WVE DEBUG


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

    if (_updateFMaxPerEvent!=0) {
//       cout << "RooGenContext::_updateFMaxPerEvent is true, updating maximum to " << _pdfClone->maxVal(_updateFMaxPerEvent) << " / " << _pdfClone->getNorm(_otherVars) << endl ;
      _maxVar->setVal(_pdfClone->maxVal(_updateFMaxPerEvent)/_pdfClone->getNorm(_otherVars)) ;
    }

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
    while((uniVar=(RooAbsArg*)_uniIter->Next())) {
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


//   cout << "RooGenContext::generateEvent(" << _pdfClone->GetName() << ") theEvent [internal @ end] = " << endl ;
//   _theEvent->Print("v") ;
//   cout << "RooGenContext::generateEvent(" << _pdfClone->GetName() << ") theEvent [argument @ end] = " << endl ;
//   theEvent.Print("v") ;

//   cout << "RooGenContext::generateEvent() the end" << endl ;
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
