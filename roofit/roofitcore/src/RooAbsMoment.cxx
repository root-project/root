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
**/

#include <RooAbsMoment.h>
#include <RooRealVar.h>

#include <Riostream.h>

#include <cmath>
#include <string>


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
