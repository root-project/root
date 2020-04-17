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
\file RooRealAnalytic.cxx
\class RooRealAnalytic
\ingroup Roofitcore

Lightweight RooAbsFunc interface adaptor that binds an analytic integral of a
RooAbsReal object (specified by a code) to a set of dependent variables.
**/


#include "RooFit.h"

#include "RooRealAnalytic.h"
#include "RooAbsReal.h"

#include <assert.h>

using namespace std;

ClassImp(RooRealAnalytic);
;


////////////////////////////////////////////////////////////////////////////////
/// Evaluate our analytic integral at the specified values of the dependents.

Double_t RooRealAnalytic::operator()(const Double_t xvector[]) const 
{
  assert(isValid());
  loadValues(xvector);  
  _ncall++ ;
  return _code ? _func->analyticalIntegralWN(_code,_nset,_rangeName?_rangeName->GetName():0):_func->getVal(_nset) ;
}
