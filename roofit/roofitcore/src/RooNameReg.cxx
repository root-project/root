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
\file RooNameReg.cxx
\class RooNameReg
\ingroup Roofitcore

RooNameReg is a registry for 'const char*' name. For each unique
name (which is not necessarily a unique pointer in the C++ standard),
a unique pointer to a TNamed object is return that can be used for
fast searches and comparisons.
**/

#include "RooFit.h"
#include "RooSentinel.h"

#include "RooNameReg.h"
#include "RooNameReg.h"
#include <iostream>
using namespace std ;

ClassImp(RooNameReg);
;

RooNameReg* RooNameReg::_instance = 0 ;


RooNameReg::RooNameReg(Int_t hashSize) : TNamed("RooNameReg","RooFit Name Registry"), _htable(hashSize) {} 

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNameReg::~RooNameReg()
{
  _list.Delete() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooNameReg::RooNameReg(const RooNameReg& other) : TNamed(other)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return reference to singleton instance

RooNameReg& RooNameReg::instance()
{
  if (_instance==0) {
    _instance = new RooNameReg(100000) ;  // there's only one of these, so we can afford to make it large
    RooSentinel::activate() ;
  }
  return *_instance ;
}


////////////////////////////////////////////////////////////////////////////////
/// Cleanup function called by atexit() handler installed by RooSentinel
/// to delete global objects on heap at end of program

void RooNameReg::cleanup()
{
  if(_instance) {
    delete _instance ;
    _instance = 0 ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return a unique TNamed pointer for given C++ string

const TNamed* RooNameReg::constPtr(const char* inStr) 
{
  // Handle null pointer case explicitly
  if (inStr==0) return 0 ;

//   cout << "RooNameReg::constPtr(inStr=" << inStr << ") _htable entries = " << _htable.entries() << endl ;

  // See if name is already registered ;
  TNamed* t = (TNamed*) _htable.find(inStr) ;
  if (t) return t ;

  // If not, register now
  t = new TNamed(inStr,inStr) ;
  _htable.add(t) ;
  _list.Add(t) ;
  
  return t ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return C++ string corresponding to given TNamed pointer

const char* RooNameReg::constStr(const TNamed* namePtr) 
{
  if (namePtr) return namePtr->GetName() ;
  return 0 ;  
}


////////////////////////////////////////////////////////////////////////////////
/// Return a unique TNamed pointer for given C++ string

const TNamed* RooNameReg::ptr(const char* stringPtr) 
{ 
  if (stringPtr==0) return 0 ;
  return instance().constPtr(stringPtr) ; 
}


////////////////////////////////////////////////////////////////////////////////
/// Return C++ string corresponding to given TNamed pointer

const char* RooNameReg::str(const TNamed* ptr) 
{ 
  if (ptr==0) return 0 ;
  return instance().constStr(ptr) ; 
}


////////////////////////////////////////////////////////////////////////////////
/// If the name is already known, return its TNamed pointer. Otherwise return 0 (don't register the name).

const TNamed* RooNameReg::known(const char* inStr)
{
  // Handle null pointer case explicitly
  if (inStr==0) return 0 ;
  if (_instance==0) return 0;
  return (const TNamed*) _instance->_htable.find(inStr) ;
}
