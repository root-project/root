/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealBinding.cc,v 1.1 2001/08/03 21:44:57 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
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
"$Id: RooRealBinding.cc,v 1.1 2001/08/03 21:44:57 david Exp $";

RooRealBinding::RooRealBinding(const RooAbsReal& func, const RooArgSet &vars) :
  RooAbsFunc(vars.getSize()), _func(&func), _vars(0)
{
  // allocate memory
  _vars= new RooAbsRealLValue*[getDimension()];
  if(0 == _vars) {
    _valid= kFALSE;
    return;
  }
  // check that all of the arguments are real valued and store them
  RooAbsArg *var(0);
  for(Int_t index= 0; index < _dimension; index++) {
    var= (RooAbsArg*)vars.At(index);
    _vars[index]= dynamic_cast<RooAbsRealLValue*>(var);
    if(0 == _vars[index]) {
      cout << "RooRealBinding: cannot bind to ";
      var->Print("1");
      _valid= kFALSE;
    }
  }
}

RooRealBinding::~RooRealBinding() {
  if(0 != _vars) delete[] _vars;
}

void RooRealBinding::loadValues(const Double_t xvector[]) const {
  for(Int_t index= 0; index < _dimension; index++) {
    _vars[index]->setVal(xvector[index]);
  }
}  

Double_t RooRealBinding::operator()(const Double_t xvector[]) const {
  assert(isValid());
  loadValues(xvector);
  return _func->getVal();
}

Double_t RooRealBinding::getMinLimit(UInt_t index) const {
  assert(isValid());
  return _vars[index]->getFitMin();
}

Double_t RooRealBinding::getMaxLimit(UInt_t index) const {
  assert(isValid());
  return _vars[index]->getFitMax();
}
