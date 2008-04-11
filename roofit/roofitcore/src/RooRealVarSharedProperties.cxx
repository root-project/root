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

// -- CLASS DESCRIPTION [AUX] --

#include "RooFit.h"
#include "RooRealVarSharedProperties.h"
#include "Riostream.h"


ClassImp(RooRealVarSharedProperties)
;


RooRealVarSharedProperties::RooRealVarSharedProperties() 
{
//   cout << "RooRealVarSharedProperties::defctor(" << this << ")" << endl ;
} 

RooRealVarSharedProperties::RooRealVarSharedProperties(const char* uuidstr) : RooSharedProperties(uuidstr)
{
//   cout << "RooRealVarSharedProperties::ctor(" << this << ")" << endl ;
} 

RooRealVarSharedProperties::RooRealVarSharedProperties(const RooRealVarSharedProperties& other) :
  RooSharedProperties(other), 
  _altBinning(other._altBinning)
{
//   cout << "RooRealVarSharedProperties::cctor(" << this << ") other = " << &other << endl ;
}



RooRealVarSharedProperties::~RooRealVarSharedProperties() 
{
//   cout << "RooRealVarSharedProperties::dtor(" << this << ")" << endl ;
  _altBinning.Delete() ;
} 


