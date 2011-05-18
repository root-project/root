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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Lightweight interface adaptor that binds a RooAbsReal object to a subset
// of its servers and present it as a simple array oriented interface.
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsRealLValue.h"
#include "RooNameReg.h"
#include "RooMsgService.h"

#include <assert.h>



ClassImp(RooRealBinding)
;


//_____________________________________________________________________________
RooRealBinding::RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset, Bool_t clipInvalid, const TNamed* rangeName) :
  RooAbsFunc(vars.getSize()), _func(&func), _vars(0), _nset(nset), _clipInvalid(clipInvalid), _xsave(0), _rangeName(rangeName)
{
  // Construct a lightweight function binding of RooAbsReal func to
  // variables 'vars'.  Use the provided nset as normalization set to
  // be passed to RooAbsReal::getVal() If rangeName is not null, use
  // the range of with that name as range associated with the
  // variables of this function binding. If clipInvalid is true,
  // values requested to the function binding that are outside the
  // defined range of the variables are clipped to fit in the defined
  // range.

  // allocate memory
  _vars= new RooAbsRealLValue*[getDimension()];
  if(0 == _vars) {
    _valid= kFALSE;
    return;
  }
  // check that all of the arguments are real valued and store them
  RooAbsArg *var = 0;
  TIterator* iter = vars.createIterator() ;
  Int_t index(0) ;
  while((var=(RooAbsArg*)iter->Next())) {
    _vars[index]= dynamic_cast<RooAbsRealLValue*>(var);
    if(0 == _vars[index]) {
      oocoutE((TObject*)0,InputArguments) << "RooRealBinding: cannot bind to " << var->GetName() << endl ;
      _valid= kFALSE;
    }
    index++ ;
  }
  delete iter ;
  _xvecValid = kTRUE ;
}


//_____________________________________________________________________________
RooRealBinding::RooRealBinding(const RooRealBinding& other, const RooArgSet* nset) :
  RooAbsFunc(other), _func(other._func), _nset(nset?nset:other._nset), _xvecValid(other._xvecValid),
  _clipInvalid(other._clipInvalid), _xsave(0), _rangeName(other._rangeName)
{
  // Construct a lightweight function binding of RooAbsReal func to
  // variables 'vars'.  Use the provided nset as normalization set to
  // be passed to RooAbsReal::getVal() If rangeName is not null, use
  // the range of with that name as range associated with the
  // variables of this function binding. If clipInvalid is true,
  // values requested to the function binding that are outside the
  // defined range of the variables are clipped to fit in the defined
  // range.

  // allocate memory
  _vars= new RooAbsRealLValue*[getDimension()];

  for(unsigned int index=0 ; index<getDimension() ; index++) {
    _vars[index]= other._vars[index] ;
  }
}


//_____________________________________________________________________________
RooRealBinding::~RooRealBinding() 
{
  // Destructor

  if(0 != _vars) delete[] _vars;
  if (_xsave) delete[] _xsave ;
}



//_____________________________________________________________________________
void RooRealBinding::saveXVec() const
{
  // Save value of all variables

  if (!_xsave) {
    _xsave = new Double_t[getDimension()] ;    
    for (UInt_t i=0 ; i<getDimension() ; i++) {
      _xsave[i] = _vars[i]->getVal() ;
    }
  }
}


//_____________________________________________________________________________
void RooRealBinding::restoreXVec() const
{
  // Restore value of all variables to previously
  // saved values by saveXVec()

  if (!_xsave) {
    return ;
  }
  for (UInt_t i=0 ; i<getDimension() ; i++) {
   _vars[i]->setVal(_xsave[i]) ;
  }
}



//_____________________________________________________________________________
void RooRealBinding::loadValues(const Double_t xvector[]) const 
{
  // Load the vector of variable values into the RooRealVars associated
  // as variables with the bound RooAbsReal function

  _xvecValid = kTRUE ;
  for(UInt_t index= 0; index < _dimension; index++) {
    if (_clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = kFALSE ;
    } else {
      _vars[index]->setVal(xvector[index],RooNameReg::instance().constStr(_rangeName));
    }
  }
}  


//_____________________________________________________________________________
Double_t RooRealBinding::operator()(const Double_t xvector[]) const 
{
  // Evaluate the bound RooAbsReal at the variable values provided in xvector

  assert(isValid());
  _ncall++ ;
  loadValues(xvector);
  //cout << getName() << "(x=" << xvector[0] << ")=" << _func->getVal(_nset) << " (nset = " << (_nset? *_nset:RooArgSet()) << ")" << endl ;
  return _xvecValid ? _func->getVal(_nset) : 0. ;
}


//_____________________________________________________________________________
Double_t RooRealBinding::getMinLimit(UInt_t index) const 
{
  // Return lower limit on i-th variable 
  assert(isValid());

  return _vars[index]->getMin(RooNameReg::str(_rangeName));
}


//_____________________________________________________________________________
Double_t RooRealBinding::getMaxLimit(UInt_t index) const 
{
  // Return upper limit on i-th variable 

  assert(isValid());
  return _vars[index]->getMax(RooNameReg::str(_rangeName));
}


//_____________________________________________________________________________
const char* RooRealBinding::getName() const 
{ 
  // Return name of function

  return _func->GetName() ; 
} 


//_____________________________________________________________________________
std::list<Double_t>* RooRealBinding::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const 
{
  return _func->plotSamplingHint(obs,xlo,xhi) ; 
}
