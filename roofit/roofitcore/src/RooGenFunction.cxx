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
// Lightweight interface adaptor that exports a RooAbsReal as a ROOT::Math::IGenFunction
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include "RooGenFunction.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <assert.h>



using namespace std;

ClassImp(RooGenFunction)
;

//_____________________________________________________________________________
RooGenFunction::RooGenFunction(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters) :
  _ftor(func,observables,parameters,observables)
{
}


//_____________________________________________________________________________
RooGenFunction::RooGenFunction(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters, const RooArgSet& nset) :
  _ftor(func,observables,parameters,nset)
{
}


//_____________________________________________________________________________
RooGenFunction::RooGenFunction(const RooGenFunction& other) :
  ROOT::Math::IGenFunction(other), _ftor(other._ftor)
{
}


//_____________________________________________________________________________
RooGenFunction::~RooGenFunction() 
{
}


//_____________________________________________________________________________
double RooGenFunction::DoEval(double x) const 
{
  return _ftor(x) ;
}


