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
\file RooConstVar.cxx
\class RooConstVar
\ingroup Roofitcore

RooConstVar represent a constant real-valued object
**/


#include "RooFit.h"

#include "Riostream.h"
#include "RooConstVar.h"

using namespace std;

ClassImp(RooConstVar);
  ;



////////////////////////////////////////////////////////////////////////////////
/// Constructor with value
///_fast = kTRUE ;

RooConstVar::RooConstVar(const char *name, const char *title, Double_t value) : 
  RooAbsReal(name,title), _value(value)
{  
  setAttribute("Constant",kTRUE) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor
///_fast = kTRUE ;

RooConstVar::RooConstVar(const RooConstVar& other, const char* name) : 
  RooAbsReal(other, name), _value(other._value)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooConstVar::~RooConstVar() 
{
}


////////////////////////////////////////////////////////////////////////////////
/// Write object contents to stream

void RooConstVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  os << _value ;
}

