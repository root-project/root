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
\file RooConvIntegrandBinding.cxx
\class RooConvIntegrandBinding
\ingroup Roofitcore

Implementation of RooAbsFunc that represent the integrand
of a generic (numeric) convolution A (x) B so that it can be
passed to a numeric integrator. This is a utility class for
RooNumConvPdf
**/

#include "RooConvIntegrandBinding.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"

#include <assert.h>

using namespace std;

ClassImp(RooConvIntegrandBinding);
;


////////////////////////////////////////////////////////////////////////////////

RooConvIntegrandBinding::RooConvIntegrandBinding(const RooAbsReal& func, const RooAbsReal& model,
                   RooAbsReal& xprime, RooAbsReal& x,
                   const RooArgSet* nset, bool clipInvalid) :

  RooAbsFunc(2), _func(&func), _model(&model), _vars(0), _nset(nset), _clipInvalid(clipInvalid)
{
  // Constructor where func and model
  //
  // 'func'  = func(xprime)
  // 'model' = model(xprime)
  //
  // and

  // 'xprime' is the RRV that should be connected to func and model
  //          (i.e. the variable that will be integrated over)
  // 'x'      is RRV that represents the value at which the convolution is calculated
  //          (this variable should _not_ be connected to func and model)
  //
  // this function returns RCBB[x',x] = f[x']*g[x-x'], i.e. the substiturion g[x'] --> g[x-x']
  // is taken care internally
  //
  // The integral of this binding over its 1st arg yields the convolution (f (x) g)[x]
  //

  // allocate memory
  _vars= new RooAbsRealLValue*[2];
  if(0 == _vars) {
    _valid= false;
    return;
  }

  // check that all of the arguments are real valued and store them
  _vars[0]= dynamic_cast<RooAbsRealLValue*>(&xprime);
  if(0 == _vars[0]) {
    oocoutE(&func,InputArguments) << "RooConvIntegrandBinding: cannot bind to ";
    xprime.Print("1");
    _valid= false;
  }

  _vars[1]= dynamic_cast<RooAbsRealLValue*>(&x);
  if(0 == _vars[1]) {
    oocoutE(&func,InputArguments) << "RooConvIntegrandBinding: cannot bind to ";
    x.Print("1");
    _valid= false;
  }

  _xvecValid = true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooConvIntegrandBinding::~RooConvIntegrandBinding()
{
  if(0 != _vars) delete[] _vars;
}


////////////////////////////////////////////////////////////////////////////////
/// Load external input values

void RooConvIntegrandBinding::loadValues(const double xvector[], bool clipInvalid) const
{
  _xvecValid = true ;
  for(UInt_t index= 0; index < _dimension; index++) {
    if (clipInvalid && !_vars[index]->isValidReal(xvector[index])) {
      _xvecValid = false ;
    } else {
      //cout << "RooConvBasBinding::loadValues[" << index << "] loading value " << xvector[index] << endl ;
      _vars[index]->setVal(xvector[index]);
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate self at given parameter values

double RooConvIntegrandBinding::operator()(const double xvector[]) const
{
  assert(isValid());
  _ncall++ ;

  // First evaluate function at x'
  loadValues(xvector);
  if (!_xvecValid) return 0 ;
  //cout << "RooConvIntegrandBinding::operator(): evaluating f(x') at x' = " << xvector[0] << endl ;
  double f_xp = _func->getVal(_nset) ;

  // Next evaluate model at x-x'
  const double xvec_tmp[2] = { xvector[1]-xvector[0] , xvector[1] } ;
  loadValues(xvec_tmp,true);
  if (!_xvecValid) return 0 ;
  double g_xmxp = _model->getVal(_nset) ;

  //cout << "RooConvIntegrandBinding::operator(): evaluating g(x-x') at x-x' = " << _vars[0]->getVal() << " = " << g_xmxp << endl ;
  //cout << "RooConvIntegrandBinding::operator(): return value = " << f_xp << " * " << g_xmxp << " = " << f_xp*g_xmxp << endl ;

  //cout << "_vars[0] = " << _vars[0]->getVal() << " _vars[1] = " << _vars[1]->getVal() << endl ;
  //cout << "_xvec[0] = " <<  xvector[0]        << " _xvec[1] = " <<  xvector[1] << endl ;

  return f_xp*g_xmxp ;
}


////////////////////////////////////////////////////////////////////////////////
/// Retrieve lower limit of i-th observable

double RooConvIntegrandBinding::getMinLimit(UInt_t index) const
{
  assert(isValid());
  return _vars[index]->getMin();
}


////////////////////////////////////////////////////////////////////////////////
/// Retrieve upper limit of i-th observable

double RooConvIntegrandBinding::getMaxLimit(UInt_t index) const
{
  assert(isValid());
  return _vars[index]->getMax();
}
