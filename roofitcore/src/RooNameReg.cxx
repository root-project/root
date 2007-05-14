/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooNameReg.cxx,v 1.5 2007/05/11 09:11:58 verkerke Exp $
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

#include "RooNameReg.h"
#include "RooNameReg.h"

ClassImp(RooNameReg)
;

RooNameReg* RooNameReg::_instance = 0 ;


RooNameReg::~RooNameReg()
{
}

RooNameReg::RooNameReg(const RooNameReg& other) : TNamed(other)
{
}

RooNameReg& RooNameReg::instance()
{
  if (_instance==0) {
    _instance = new RooNameReg ;
  }
  return *_instance ;
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
  
  return t ;
}


const char* RooNameReg::constStr(const TNamed* namePtr) 
{
  if (namePtr) return namePtr->GetName() ;
  return 0 ;  
}
