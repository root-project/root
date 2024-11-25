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
\file RooEfficiency.cxx
\class RooEfficiency
\ingroup Roofitcore

A PDF helper class to fit efficiencies parameterized
by a supplied function F.

Given a dataset with a category C that determines if a given
event is accepted or rejected for the efficiency to be measured,
this class evaluates as F if C is 'accept' and as (1-F) if
C is 'reject'. Values of F below 0 and above 1 are clipped.
F may have an arbitrary number of dependents and parameters
**/

#include "RooEfficiency.h"

#include "RooStreamParser.h"
#include "RooArgList.h"

#include <RooFit/Detail/MathFuncs.h>

#include "TError.h"

ClassImp(RooEfficiency);


////////////////////////////////////////////////////////////////////////////////
/// Construct an N+1 dimensional efficiency p.d.f from an N-dimensional efficiency
/// function and a category cat with two states (0,1) that indicate if a given
/// event should be counted as rejected or accepted respectively

RooEfficiency::RooEfficiency(const char *name, const char *title, const RooAbsReal& effFunc, const RooAbsCategory& cat, const char* sigCatName) :
  RooAbsPdf(name,title),
  _cat("cat","Signal/Background category",this,(RooAbsCategory&)cat),
  _effFunc("effFunc","Efficiency modeling function",this,(RooAbsReal&)effFunc),
  _sigCatName(sigCatName)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooEfficiency::RooEfficiency(const RooEfficiency& other, const char* name) :
  RooAbsPdf(other, name),
  _cat("cat",this,other._cat),
  _effFunc("effFunc",this,other._effFunc),
  _sigCatName(other._sigCatName)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate the raw value of this p.d.f which is the effFunc
/// value if cat==1 and it is (1-effFunc) if cat==0

double RooEfficiency::evaluate() const
{
   const int sigCatIndex = _cat->lookupIndex(_sigCatName.Data());
   return RooFit::Detail::MathFuncs::efficiency(_effFunc, _cat, sigCatIndex);
}

int RooEfficiency::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, _cat) ? 1 : 0;
}

double RooEfficiency::analyticalIntegral(int /*code*/, const char * /*rangeName*/) const
{
   return 1.0;
}
