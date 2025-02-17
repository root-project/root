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
\file RooFunctor.cxx
\class RooFunctor
\ingroup Roofitcore

Lightweight interface adaptor that exports a RooAbsPdf as a functor.
**/

#include "Riostream.h"

#include "RooFunctor.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <cassert>



////////////////////////////////////////////////////////////////////////////////

RooFunctor::RooFunctor(const RooAbsFunc &func)
   : _binding(const_cast<RooAbsFunc *>(&func)), _x(func.getDimension()), _nobs(func.getDimension())
{
}

////////////////////////////////////////////////////////////////////////////////
/// Store list of observables

RooFunctor::RooFunctor(const RooAbsReal &func, const RooArgList &observables, const RooArgList &parameters)
   : RooFunctor{func, observables, parameters, observables}
{
}

////////////////////////////////////////////////////////////////////////////////
/// Store normalization set

RooFunctor::RooFunctor(const RooAbsReal &func, const RooArgList &observables, const RooArgList &parameters,
                       const RooArgSet &nset)
   : _npar(parameters.size()), _nobs(observables.size())
{
  _nset.add(nset) ;

  // Make list of all variables to be bound
  RooArgList allVars(observables) ;
  allVars.add(parameters) ;

  // Create RooFit function binding
  _ownedBinding = std::make_unique<RooRealBinding>(func,allVars,&_nset,false,nullptr) ;

  // Allocate transfer array
  _x.resize(allVars.size());
}

////////////////////////////////////////////////////////////////////////////////

RooFunctor::RooFunctor(const RooFunctor& other) :
  _nset(other._nset),
  _binding{other._binding},
  _npar(other._npar),
  _nobs(other._nobs)
{
  if (other._ownedBinding) {
    _ownedBinding = std::make_unique<RooRealBinding>(static_cast<RooRealBinding&>(*other._ownedBinding),&_nset);
  }
  _x.resize(_nobs + _npar);
}

RooFunctor::~RooFunctor() = default;

////////////////////////////////////////////////////////////////////////////////

double RooFunctor::eval(const double *x) const
{
  return binding()(x) ;
}

////////////////////////////////////////////////////////////////////////////////

double RooFunctor::eval(double x) const
{
  return binding()(&x) ;
}

////////////////////////////////////////////////////////////////////////////////

double RooFunctor::eval(const double *x, const double *p) const
{
  for (int i=0 ; i<_nobs ; i++) {
    _x[i] = x[i] ;
  }
  for (int i=0 ; i<_npar ; i++) {
    _x[i+_nobs] = p[i] ;
  }
  return binding()(_x.data());
}
