/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConstVar.cc,v 1.4 2003/05/14 05:33:29 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooConstVar represent a constant real-valued object


#include "RooFitCore/RooConstVar.hh"

ClassImp(RooConstVar)
  ;


RooConstVar::RooConstVar(const char *name, const char *title, Double_t value) : 
  RooAbsReal(name,title), _value(value)
{  
  setAttribute("Constant",kTRUE) ;
}


RooConstVar::RooConstVar(const RooConstVar& other, const char* name) : 
  RooAbsReal(other, name), _value(other._value)
{
  // Copy constructor
}


RooConstVar::~RooConstVar() 
{
}


void RooConstVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream
  os << _value ;
}

