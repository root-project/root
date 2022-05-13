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
#ifndef ROO_FUNCTOR
#define ROO_FUNCTOR

#include "RooArgSet.h"
#include "RooAbsReal.h"

class RooAbsFunc ;
class RooAbsPdf ;

class RooFunctor {

public:
  RooFunctor(const RooAbsFunc& func) ;
  RooFunctor(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters) ;
  RooFunctor(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters, const RooArgSet& nset) ;
  RooFunctor(const RooFunctor&) ;
  virtual ~RooFunctor() ;

  Int_t nObs() const {
    // Number of observables
    return _nobs ;
  }
  Int_t nPar() const {
    // Number of parameters;
    return _npar ;
  }

  double operator()(double x) const { return eval(x) ; }
  double operator()(const double* x, const double* p) const { return eval(x,p) ; }
  double operator()(const double* x) const { return eval(x) ; }

  double eval(const double* /*x*/, const double* /*p*/) const ;
  double eval(const double* /*x*/) const ;
  double eval(double  /*x*/) const ;

  RooAbsFunc& binding() { return *_binding ; }

protected:

  bool          _ownBinding ; ///< Do we own the binding function
  RooArgSet       _nset ;       ///< Normalization observables
  RooAbsFunc*     _binding ;    ///< Function binding
  double*       _x ;          ///<! Transfer array ;
  Int_t           _npar ;       ///<! Number of parameters ;
  Int_t           _nobs ;       ///<! Number of observables ;

  ClassDef(RooFunctor,0) // Export RooAbsReal as functor
};

#endif

