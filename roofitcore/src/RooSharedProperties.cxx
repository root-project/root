/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSharedProperties.cc,v 1.3 2006/12/08 10:13:37 wverkerke Exp $
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
#include "RooSharedProperties.h"
#include "RooTrace.h"

#include <iostream>
using std::cout ;
using std::endl ;

ClassImp(RooSharedProperties)
;


RooSharedProperties::RooSharedProperties() 
{
  RooTrace::create(this) ;
} 

RooSharedProperties::RooSharedProperties(const char* uuidstr) : _uuid(uuidstr) 
{
  RooTrace::create(this) ;
} 


RooSharedProperties::~RooSharedProperties() 
{
  RooTrace::destroy(this) ;
} 


Bool_t RooSharedProperties::operator==(const RooSharedProperties& other) 
{
  // Forward comparison to Unique UID component
  return (_uuid==other._uuid) ;
}

void RooSharedProperties::Print(Option_t* /*opts*/) const 
{
  cout << "RooSharedProperties(" << this << ") UUID = " << _uuid.AsString() << endl ;
}
