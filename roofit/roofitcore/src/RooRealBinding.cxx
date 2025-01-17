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

#include "RooRealBinding.h"

#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsRealLValue.h"
#include "RooNameReg.h"
#include "RooMsgService.h"

#include <cassert>

using std::endl;

ClassImp(RooRealBinding);


////////////////////////////////////////////////////////////////////////////////
/// Construct a lightweight function binding of RooAbsReal func to
/// variables 'vars'.  Use the provided nset as normalization set to
/// be passed to RooAbsReal::getVal() If rangeName is not null, use
/// the range of with that name as range associated with the
/// variables of this function binding. If clipInvalid is true,
/// values requested to the function binding that are outside the
/// defined range of the variables are clipped to fit in the defined
/// range.

RooRealBinding::RooRealBinding(const RooAbsReal &func, const RooArgSet &vars, const RooArgSet *nset, bool clipInvalid,
                               const TNamed *rangeName)
   : RooAbsFunc(vars.size()), _func(&func), _nset(nset), _clipInvalid(clipInvalid), _rangeName(rangeName), _funcSave(0)
{
  // check that all of the arguments are real valued and store them
  for (unsigned int index=0; index < vars.size(); ++index) {
    RooAbsArg* var = vars[index];
    _vars.push_back(dynamic_cast<RooAbsRealLValue*>(var));
    if(_vars.back() == nullptr) {
      oocoutE(nullptr,InputArguments) << "RooRealBinding: cannot bind to " << var->GetName()
          << ". Variables need to be assignable, e.g. instances of RooRealVar." << std::endl ;
      _valid= false;
    }
    if (!_func->dependsOn(*_vars[index])) {
      oocoutW(nullptr, InputArguments) << "RooRealBinding: The function " << func.GetName() << " does not depend on the parameter " << _vars[index]->GetName()
          << ". Note that passing copies of the parameters is not supported." << std::endl;
    }
  }

  _xvecValid = true ;
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
  RooAbsFunc(other), _func(other._func), _vars(other._vars), _nset(nset?nset:other._nset), _xvecValid(other._xvecValid),
  _clipInvalid(other._clipInvalid), _rangeName(other._rangeName), _funcSave(other._funcSave)
{

}


RooRealBinding::~RooRealBinding() = default;


////////////////////////////////////////////////////////////////////////////////
/// Save value of all variables

void RooRealBinding::saveXVec() const
{
  if (_xsave.empty()) {
    _xsave.resize(getDimension());
    std::unique_ptr<RooArgSet> comps{_func->getComponents()};
    for (auto* arg : dynamic_range_cast<RooAbsArg*>(*comps)) {
      if (arg) {
        _compList.push_back(static_cast<RooAbsReal*>(arg)) ;
        _compSave.push_back(0.0) ;
      }
    }
  }
  _funcSave = _func->_value ;

  // Save components
  auto ci = _compList.begin() ;
  auto si = _compSave.begin() ;
  while(ci != _compList.end()) {
    *si = (*ci)->_value ;
    ++si;
    ++ci;
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
  if (_xsave.empty()) {
    return ;
  }
  _func->_value = _funcSave ;

  // Restore components
  auto ci = _compList.begin() ;
  auto si = _compSave.begin() ;
  while (ci != _compList.end()) {
    (*ci)->_value = *si ;
    ++ci;
    ++si;
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
void RooRealBinding::loadValues(const double xvector[]) const
{
  _xvecValid = true ;
  const char* range = RooNameReg::str(_rangeName) ;
  for(UInt_t index= 0; index < _dimension; index++) {
    if (_clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = false ;
    } else {
      _vars[index]->setVal(xvector[index],range);
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate the bound RooAbsReal at the variable values provided in xvector

double RooRealBinding::operator()(const double xvector[]) const
{
  assert(isValid());
  _ncall++ ;
  loadValues(xvector);
  return _xvecValid ? _func->getVal(_nset) : 0. ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return lower limit on i-th variable

double RooRealBinding::getMinLimit(UInt_t index) const
{
  assert(isValid());

  return _vars[index]->getMin(RooNameReg::str(_rangeName));
}


////////////////////////////////////////////////////////////////////////////////
/// Return upper limit on i-th variable

double RooRealBinding::getMaxLimit(UInt_t index) const
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

std::list<double>* RooRealBinding::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  return _func->plotSamplingHint(obs,xlo,xhi) ;
}


////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooRealBinding::binBoundaries(Int_t index) const
{
  return _func->binBoundaries(*_vars[index],getMinLimit(index),getMaxLimit(index));
}
