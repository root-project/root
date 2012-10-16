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
// Lightweight interface adaptor that exports a RooAbsPdf as a functor
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooFunctor.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <assert.h>



using namespace std;

ClassImp(RooFunctor)
;


//_____________________________________________________________________________
RooFunctor::RooFunctor(const RooAbsFunc& func)
{
  _ownBinding = kFALSE ;

  _x = new Double_t[func.getDimension()] ; 

  _nobs = func.getDimension() ;
  _npar = 0 ;
  _binding = (RooAbsFunc*) &func ;
}



//_____________________________________________________________________________
RooFunctor::RooFunctor(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters) 
{
  // Store list of observables
  _nset.add(observables) ;

  // Make list of all variables to be bound
  RooArgList allVars(observables) ;
  allVars.add(parameters) ;

  // Create RooFit function binding
  _binding = new RooRealBinding(func,allVars,&_nset,kFALSE,0) ;
  _ownBinding = kTRUE ;

  // Allocate transfer array
  _x = new Double_t[allVars.getSize()] ; 
  _nobs = observables.getSize() ;
  _npar = parameters.getSize() ;
}


//_____________________________________________________________________________
RooFunctor::RooFunctor(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters, const RooArgSet& nset) 
{
  // Store normalization set
  _nset.add(nset) ;

  // Make list of all variables to be bound
  RooArgList allVars(observables) ;
  allVars.add(parameters) ;

  // Create RooFit function binding
  _binding = new RooRealBinding(func,allVars,&_nset,kFALSE,0) ;
  _ownBinding = kTRUE ;

  // Allocate transfer array
  _x = new Double_t[allVars.getSize()] ; 
  _nobs = observables.getSize() ;
  _npar = parameters.getSize() ;
}



//_____________________________________________________________________________
RooFunctor::RooFunctor(const RooFunctor& other) :
  _ownBinding(other._ownBinding),
  _nset(other._nset),
  _binding(0),
  _npar(other._npar),
  _nobs(other._nobs)
{
  if (other._ownBinding) {
    _binding = new RooRealBinding((RooRealBinding&)*other._binding,&_nset) ;
  } else {
    _binding = other._binding ;
  }
  _x = new Double_t[_nobs+_npar] ;
}




//_____________________________________________________________________________
RooFunctor::~RooFunctor() 
{
  // Destructor
  if (_ownBinding) delete _binding ; 
  delete[] _x ;
}



//_____________________________________________________________________________
Double_t RooFunctor::eval(const Double_t *x) const
{
  return (*_binding)(x) ;
}

//_____________________________________________________________________________
Double_t RooFunctor::eval(Double_t x) const
{
  return (*_binding)(&x) ;
}

//_____________________________________________________________________________
Double_t RooFunctor::eval(const Double_t *x, const Double_t *p) const
{
  for (int i=0 ; i<_nobs ; i++) { 
    _x[i] = x[i] ; 
  }
  for (int i=0 ; i<_npar ; i++) { 
    _x[i+_nobs] = p[i] ; 
  }
  return (*_binding)(_x) ;
}
