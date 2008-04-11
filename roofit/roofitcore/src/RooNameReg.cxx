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

// -- CLASS DESCRIPTION [MISC] --
// RooNameReg is a registry for 'const char*' name. For each unique
// name (which is not necessarily a unique pointer in the C++ standard)
// a unique pointer to a TNamed object is return that can be used for
// fast searches and comparisons.

#include "RooFit.h"
#include "RooSentinel.h"

#include "RooNameReg.h"
#include "RooNameReg.h"

ClassImp(RooNameReg)
;

RooNameReg* RooNameReg::_instance = 0 ;


RooNameReg::~RooNameReg()
{
  _list.Delete() ;
}

RooNameReg::RooNameReg(const RooNameReg& other) : TNamed(other)
{
}

RooNameReg& RooNameReg::instance()
{
  if (_instance==0) {
    _instance = new RooNameReg ;
    RooSentinel::activate() ;
  }
  return *_instance ;
}

void RooNameReg::cleanup()
{
  if(_instance) {
    delete _instance ;
    _instance = 0 ;
  }
}


const TNamed* RooNameReg::constPtr(const char* str) {

  // Handle null pointer case explicitly
  if (str==0) return 0 ;

  // See if name is already registered ;
  TNamed* t = (TNamed*) _htable.find(str) ;
  if (t) return t ;

  // If not, register now
  t = new TNamed(str,str) ;
  _htable.add(t) ;
  _list.Add(t) ;
  
  return t ;
}


const char* RooNameReg::constStr(const TNamed* namePtr) 
{
  if (namePtr) return namePtr->GetName() ;
  return 0 ;  
}

const TNamed* RooNameReg::ptr(const char* stringPtr) 
{ 
  return instance().constPtr(stringPtr) ; 
}

const char* RooNameReg::str(const TNamed* ptr) 
{ 
  return instance().constStr(ptr) ; 
}
