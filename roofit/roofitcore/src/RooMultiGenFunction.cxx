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
\file RooMultiGenFunction.cxx
\class RooMultiGenFunction
\ingroup Roofitcore

Lightweight interface adaptor that exports a RooAbsReal as a ROOT::Math::IMultiGenFunction
**/


#include "RooFit.h"
#include "Riostream.h"

#include "RooMultiGenFunction.h"
#include "RooRealBinding.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"

#include <assert.h>



using namespace std;

ClassImp(RooMultiGenFunction);
;


////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction::RooMultiGenFunction(const RooAbsFunc& func) :
  _ftor(func)
{
}



////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction::RooMultiGenFunction(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters) :
  _ftor(func,observables,parameters)
{
}


////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction::RooMultiGenFunction(const RooAbsReal& func, const RooArgList& observables, const RooArgList& parameters, const RooArgSet& nset) :
  _ftor(func,observables,parameters,nset)
{
}


////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction::RooMultiGenFunction(const RooMultiGenFunction& other) :
  ROOT::Math::IMultiGenFunction(other), _ftor(other._ftor)
{
}


////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction::~RooMultiGenFunction()
{
}


////////////////////////////////////////////////////////////////////////////////

double RooMultiGenFunction::DoEval(const double* x) const
{
  return _ftor(x) ;
}



