 /***************************************************************************** 
  * Project: RooFit                                                           * 
  *                                                                           * 
  * Copyright (c) 2000-2005, Regents of the University of California          * 
  *                          and Stanford University. All rights reserved.    * 
  *                                                                           * 
  * Redistribution and use in source and binary forms,                        * 
  * with or without modification, are permitted according to the terms        * 
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             * 
  *****************************************************************************/ 

 // -- CLASS DESCRIPTION [PDF] -- 
 // Your description goes here... 

#include <iostream> 

#include "RooProfileLL.h" 
#include "RooAbsReal.h" 
#include "RooMinuit.h"
#include "RooMsgService.h"
#include "RooRealVar.h"

ClassImp(RooProfileLL) 

RooProfileLL::RooProfileLL(const char *name, const char *title, 
			   RooAbsReal& nll, const RooArgSet& observables) :
  RooAbsReal(name,title), 
  _nll("nll","-log(L) function",this,nll),
  _obs("obs","observables",this),
  _par("par","parameters",this,kFALSE,kFALSE),
  _minuit(0),
  _absMinValid(kFALSE),
  _absMin(0)
{ 
  // Determine actual parameters and observables
  RooArgSet* actualObs = nll.getObservables(observables) ;
  RooArgSet* actualPars = nll.getParameters(observables) ;

  _obs.add(*actualObs) ;
  _par.add(*actualPars) ;

  delete actualObs ;
  delete actualPars ;

  _piter = _par.createIterator() ;
  _oiter = _obs.createIterator() ;
} 


RooProfileLL::RooProfileLL(const RooProfileLL& other, const char* name) :  
  RooAbsReal(other,name), 
  _nll("nll",this,other._nll),
  _obs("obs",this,other._obs),
  _par("par",this,other._par),
  _minuit(0),
  _absMinValid(kFALSE),
  _absMin(0),
  _paramFixed(other._paramFixed)
{ 
  _piter = _par.createIterator() ;
  _oiter = _obs.createIterator() ;
} 


RooProfileLL::~RooProfileLL()
{
  // Delete instance of minuit if it was ever instantiated
  if (_minuit) {
    delete _minuit ;
  }

  delete _piter ;
  delete _oiter ;
}



Double_t RooProfileLL::evaluate() const 
{ 
  // Instantiate minuit if we haven't done that already
  if (!_minuit) {
    coutI("Minimization") << "RooProfileLL::evaluate(" << GetName() << ") Creating instance of MINUIT" << endl ;
    _minuit = new RooMinuit(const_cast<RooAbsReal&>(_nll.arg())) ;
    _minuit->setPrintLevel(-999) ;
    _minuit->setNoWarn() ;
    // _minuit->setVerbose(0) ;
  }

  // Check if constant status of any of the parameters have changed
  if (_absMinValid) {
    _piter->Reset() ;
    RooAbsArg* par ;
    while((par=(RooAbsArg*)_piter->Next())) {
      if (_paramFixed[par->GetName()] != par->isConstant()) {
	cxcoutI("Minimization") << "RooProfileLL::evaluate(" << GetName() << ") constant status of parameter " << par->GetName() << " has changed from " 
				<< (_paramFixed[par->GetName()]?"fixed":"floating") << " to " << (par->isConstant()?"fixed":"floating") 
				<< ", recalculating absolute minimum" << endl ;
	_absMinValid = kFALSE ;
	break ;
      }
    }
  }

  // Save current value of observables
  RooArgSet* obsSetOrig = (RooArgSet*) _obs.snapshot() ;

  // If we don't have the absolute minimum w.r.t all observables, calculate that first
  if (!_absMinValid) {
    
    cxcoutI("Minimization") << "RooProfileLL::evaluate(" << GetName() << ") determining minimum likelihood for current configurations w.r.t all observable" << endl ;
    

    // Find minimum with all observables floating
    const_cast<RooSetProxy&>(_obs).setAttribAll("Constant",kFALSE) ;  
    _minuit->migrad() ;

    // Save value and remember
    _absMin = _nll ;
    _absMinValid = kTRUE ;

    // Save constant status of all parameters
    _piter->Reset() ;
    RooAbsArg* par ;
    while((par=(RooAbsArg*)_piter->Next())) {
      _paramFixed[par->GetName()] = par->isConstant() ;
    }
    
    if (dologI("Minimization")) {
      cxcoutI("Minimization") << "RooProfileLL::evaluate(" << GetName() << ") minimum found at (" ;

      RooAbsReal* arg ;
      Bool_t first=kTRUE ;
      _oiter->Reset() ;
      while ((arg=(RooAbsReal*)_oiter->Next())) {
	ccxcoutI("Minimization") << (first?"":", ") << arg->GetName() << "=" << arg->getVal() ;	
	first=kFALSE ;
      }      
      ccxcoutI("Minimization") << ")" << endl ;            
    }
    
  }
  
  // Set all observables constant in the minimization
  const_cast<RooSetProxy&>(_obs).setAttribAll("Constant",kTRUE) ;  
  cout << "." ; cout.flush() ;
  _minuit->migrad() ;

  // Restore original values and constant status of observables
  TIterator* iter = obsSetOrig->createIterator() ;
  RooRealVar* var ;
  while((var=(RooRealVar*)iter->Next())) {
    RooRealVar* target = (RooRealVar*) _obs.find(var->GetName()) ;
    target->setVal(var->getVal()) ;
    target->setConstant(var->isConstant()) ;
  }
  delete iter ;
  delete obsSetOrig ;

  return _nll - _absMin ; 
} 



