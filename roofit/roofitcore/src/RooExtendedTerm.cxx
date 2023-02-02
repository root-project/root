/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id$
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
\file RooExtendedTerm.cxx
\class RooExtendedTerm
\ingroup Roofitcore

RooExtendedTerm is a p.d.f with no observables that only introduces
an extended ML term for a given number of expected events term when an extended ML
is constructed.
**/

#include "RooExtendedTerm.h"

using namespace std;

ClassImp(RooExtendedTerm);
;


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooExtendedTerm::RooExtendedTerm()
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor. An ExtendedTerm has no observables, it only introduces an extended
/// ML term with the given number of expected events when an extended ML is constructed
/// from this p.d.f.

RooExtendedTerm::RooExtendedTerm(const char *name, const char *title, const RooAbsReal& n) :
  RooAbsPdf(name,title),
  _n("n","Nexpected",this,(RooAbsReal&)n)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooExtendedTerm::RooExtendedTerm(const RooExtendedTerm& other, const char* name) :
  RooAbsPdf(other,name),
  _n("n",this,other._n)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooExtendedTerm::~RooExtendedTerm()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return number of expected events from associated event count variable

double RooExtendedTerm::expectedEvents(const RooArgSet* /*nset*/) const
{
  return _n ;
}



