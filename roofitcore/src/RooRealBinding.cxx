/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealBinding.cc,v 1.10 2003/05/09 20:48:23 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// Lightweight interface adaptor that binds a RooAbsReal object to a subset
// of its servers.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooRealBinding.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsRealLValue.hh"

#include <assert.h>

ClassImp(RooRealBinding)
;

RooRealBinding::RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset, Bool_t clipInvalid) :
  RooAbsFunc(vars.getSize()), _func(&func), _vars(0), _nset(nset), _clipInvalid(clipInvalid)
{
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
  while(var=(RooAbsArg*)iter->Next()) {
    _vars[index]= dynamic_cast<RooAbsRealLValue*>(var);
    if(0 == _vars[index]) {
      cout << "RooRealBinding: cannot bind to ";
      var->Print("1");
      _valid= kFALSE;
    }
    index++ ;
  }
  delete iter ;
}

RooRealBinding::~RooRealBinding() {
  if(0 != _vars) delete[] _vars;
}

void RooRealBinding::loadValues(const Double_t xvector[]) const {
  _xvecValid = kTRUE ;
  for(UInt_t index= 0; index < _dimension; index++) {
    if (_clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = kFALSE ;
    } else {
      _vars[index]->setVal(xvector[index]);
    }
  }
}  

Double_t RooRealBinding::operator()(const Double_t xvector[]) const {
  assert(isValid());
  _ncall++ ;
  loadValues(xvector);
  return _xvecValid ? _func->getVal(_nset) : 0. ;
}

Double_t RooRealBinding::getMinLimit(UInt_t index) const {
  assert(isValid());
  return _vars[index]->getFitMin();
}

Double_t RooRealBinding::getMaxLimit(UInt_t index) const {
  assert(isValid());
  return _vars[index]->getFitMax();
}
