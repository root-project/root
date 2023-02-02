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
\file RooAbsMoment.cxx
\class RooAbsMoment
\ingroup Roofitcore

RooAbsMoment represents the first, second, or third order derivative
of any RooAbsReal as calculated (numerically) by the MathCore Richardson
derivator class.
**/

#include "Riostream.h"
#include <math.h>

#include "RooAbsMoment.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooFunctor.h"
#include "RooFormulaVar.h"
#include "RooGlobalFunc.h"
#include "RooConstVar.h"
#include "RooRealIntegral.h"
#include <string>
using namespace std ;


ClassImp(RooAbsMoment);
;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsMoment::RooAbsMoment() : _order(1), _takeRoot(false)
{
}



////////////////////////////////////////////////////////////////////////////////

RooAbsMoment::RooAbsMoment(const char* name, const char* title, RooAbsReal& func, RooRealVar& x, Int_t orderIn, bool takeRoot) :
  RooAbsReal(name, title),
  _order(orderIn),
  _takeRoot(takeRoot),
  _nset("nset","nset",this,false,false),
  _func("function","function",this,func,false,false),
  _x("x","x",this,x,false,false),
  _mean("!mean","!mean",this,false,false)
{
}


////////////////////////////////////////////////////////////////////////////////

RooAbsMoment::RooAbsMoment(const RooAbsMoment& other, const char* name) :
  RooAbsReal(other, name),
  _order(other._order),
  _takeRoot(other._takeRoot),
  _nset("nset",this,other._nset),
  _func("function",this,other._func),
  _x("x",this,other._x),
  _mean("!mean","!mean",this,false,false)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsMoment::~RooAbsMoment()
{
}



