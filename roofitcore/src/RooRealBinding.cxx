/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealBinding.cc,v 1.5 2001/10/08 05:20:20 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
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

static const char rcsid[] =
"$Id: RooRealBinding.cc,v 1.5 2001/10/08 05:20:20 verkerke Exp $";

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
  RooAbsArg *var(0);
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
  for(Int_t index= 0; index < _dimension; index++) {
    if (_clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = kFALSE ;
    } else {
      _vars[index]->setVal(xvector[index]);
    }
  }
}  

Double_t RooRealBinding::operator()(const Double_t xvector[]) const {
  assert(isValid());
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
