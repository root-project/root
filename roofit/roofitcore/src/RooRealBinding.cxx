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
\file RooRealBinding.cxx
\class RooRealBinding
\ingroup Roofitcore

Lightweight interface adaptor that binds a RooAbsReal object to a subset
of its servers and present it as a simple array oriented interface.
**/


#include "RooFit.h"
#include "Riostream.h"

#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsRealLValue.h"
#include "RooNameReg.h"
#include "RooMsgService.h"

#include <assert.h>



using namespace std;

ClassImp(RooRealBinding);
;


////////////////////////////////////////////////////////////////////////////////
/// Construct a lightweight function binding of RooAbsReal func to
/// variables 'vars'.  Use the provided nset as normalization set to
/// be passed to RooAbsReal::getVal() If rangeName is not null, use
/// the range of with that name as range associated with the
/// variables of this function binding. If clipInvalid is true,
/// values requested to the function binding that are outside the
/// defined range of the variables are clipped to fit in the defined
/// range.

RooRealBinding::RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset, Bool_t clipInvalid, const TNamed* rangeName) :
  RooAbsFunc(vars.getSize()), _func(&func), _vars(0), _nset(nset), _clipInvalid(clipInvalid), _xsave(0), _rangeName(rangeName), _funcSave(0)
{
  // allocate memory
  _vars= new RooAbsRealLValue*[getDimension()];
  if(0 == _vars) {
    _valid= kFALSE;
    return;
  }
  // check that all of the arguments are real valued and store them
  for (unsigned int index=0; index < vars.size(); ++index) {
    RooAbsArg* var = vars[index];
    _vars[index]= dynamic_cast<RooAbsRealLValue*>(var);
    if(0 == _vars[index]) {
      oocoutE((TObject*)0,InputArguments) << "RooRealBinding: cannot bind to " << var->GetName() << endl ;
      _valid= kFALSE;
    }
  }

  _xvecValid = kTRUE ;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct a lightweight function binding of RooAbsReal func to
/// variables 'vars'.  Use the provided nset as normalization set to
/// be passed to RooAbsReal::getVal() If rangeName is not null, use
/// the range of with that name as range associated with the
/// variables of this function binding. If clipInvalid is true,
/// values requested to the function binding that are outside the
/// defined range of the variables are clipped to fit in the defined
/// range.

RooRealBinding::RooRealBinding(const RooRealBinding& other, const RooArgSet* nset) :
  RooAbsFunc(other), _func(other._func), _nset(nset?nset:other._nset), _xvecValid(other._xvecValid),
  _clipInvalid(other._clipInvalid), _xsave(0), _rangeName(other._rangeName), _funcSave(other._funcSave)
{
  // allocate memory
  _vars= new RooAbsRealLValue*[getDimension()];

  for(unsigned int index=0 ; index<getDimension() ; index++) {
    _vars[index]= other._vars[index] ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooRealBinding::~RooRealBinding() 
{
  if(0 != _vars) delete[] _vars;
  if (_xsave) delete[] _xsave ;
}



////////////////////////////////////////////////////////////////////////////////
/// Save value of all variables

void RooRealBinding::saveXVec() const
{
  if (!_xsave) {
    _xsave = new Double_t[getDimension()] ;    
    RooArgSet* comps = _func->getComponents() ;
    RooFIter iter = comps->fwdIterator() ;
    RooAbsArg* arg ;
    while ((arg=iter.next())) {
      if (dynamic_cast<RooAbsReal*>(arg)) {
	_compList.push_back((RooAbsReal*)(arg)) ;
	_compSave.push_back(0) ;
      }
    }
    delete comps ;
  }
  _funcSave = _func->_value ;

  // Save components
  list<RooAbsReal*>::iterator ci = _compList.begin() ;
  list<Double_t>::iterator si = _compSave.begin() ;
  while(ci!=_compList.end()) {
    *si = (*ci)->_value ;
    ++si ; ++ci ;
  }
  
  for (UInt_t i=0 ; i<getDimension() ; i++) {
    _xsave[i] = _vars[i]->getVal() ;
  } 
}

////////////////////////////////////////////////////////////////////////////////
/// Restore value of all variables to previously
/// saved values by saveXVec()

void RooRealBinding::restoreXVec() const
{
  if (!_xsave) {
    return ;
  }
  _func->_value = _funcSave ;

  // Restore components
  list<RooAbsReal*>::iterator ci = _compList.begin() ;
  list<Double_t>::iterator si = _compSave.begin() ;
  while (ci!=_compList.end()) {
    (*ci)->_value = *si ;
    ++ci ; ++si ;
  }

  for (UInt_t i=0 ; i<getDimension() ; i++) {
   _vars[i]->setVal(_xsave[i]) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Load the vector of variable values into the RooRealVars associated
/// as variables with the bound RooAbsReal function.
/// \warning This will load as many values as the dimensionality of the function
/// requires. The size of `xvector` is not checked.
void RooRealBinding::loadValues(const Double_t xvector[]) const 
{
  _xvecValid = kTRUE ;
  const char* range = RooNameReg::instance().constStr(_rangeName) ;
  for(UInt_t index= 0; index < _dimension; index++) {
    if (_clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = kFALSE ;
    } else {
      _vars[index]->setVal(xvector[index],range);
    }
  }

}  


////////////////////////////////////////////////////////////////////////////////
/// Evaluate the bound RooAbsReal at the variable values provided in xvector

Double_t RooRealBinding::operator()(const Double_t xvector[]) const 
{
  assert(isValid());
  _ncall++ ;
  loadValues(xvector);
  //cout << getName() << "(x=" << xvector[0] << ")=" << _func->getVal(_nset) << " (nset = " << (_nset? *_nset:RooArgSet()) << ")" << endl ;
  return _xvecValid ? _func->getVal(_nset) : 0. ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return lower limit on i-th variable 

Double_t RooRealBinding::getMinLimit(UInt_t index) const 
{
  assert(isValid());

  return _vars[index]->getMin(RooNameReg::str(_rangeName));
}


////////////////////////////////////////////////////////////////////////////////
/// Return upper limit on i-th variable 

Double_t RooRealBinding::getMaxLimit(UInt_t index) const 
{
  assert(isValid());
  return _vars[index]->getMax(RooNameReg::str(_rangeName));
}


////////////////////////////////////////////////////////////////////////////////
/// Return name of function

const char* RooRealBinding::getName() const 
{ 
  return _func->GetName() ; 
} 


////////////////////////////////////////////////////////////////////////////////

std::list<Double_t>* RooRealBinding::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const 
{
  return _func->plotSamplingHint(obs,xlo,xhi) ; 
}


////////////////////////////////////////////////////////////////////////////////

std::list<Double_t>* RooRealBinding::binBoundaries(Int_t index) const
{
  return _func->binBoundaries(*_vars[index],getMinLimit(index),getMaxLimit(index));
}
