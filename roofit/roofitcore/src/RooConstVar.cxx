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
// RooConstVar represent a constant real-valued object
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "RooConstVar.h"

ClassImp(RooConstVar)
  ;



//_____________________________________________________________________________
RooConstVar::RooConstVar(const char *name, const char *title, Double_t value) : 
  RooAbsReal(name,title), _value(value)
{  
  // Constructor with value

  setAttribute("Constant",kTRUE) ;
}



//_____________________________________________________________________________
RooConstVar::RooConstVar(const RooConstVar& other, const char* name) : 
  RooAbsReal(other, name), _value(other._value)
{

  // Copy constructor
}



//_____________________________________________________________________________
RooConstVar::~RooConstVar() 
{
  // Destructor
}



//_____________________________________________________________________________
Double_t RooConstVar::getVal(const RooArgSet*) const 
{ 
  // Return value
  return _value ; 
}



//_____________________________________________________________________________
void RooConstVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  // Write object contents to stream
  os << _value ;
}

