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

RooNameReg is a registry for `const char*` names. For each unique
name (which is not necessarily a unique pointer in the C++ standard),
a unique pointer to a TNamed object is returned that can be used for
fast searches and comparisons.
**/

#include "RooNameReg.h"

#include <iostream>
#include <memory>
using namespace std ;


RooNameReg::RooNameReg() :
    TNamed("RooNameReg","RooFit Name Registry")
{}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooNameReg::~RooNameReg()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return reference to singleton instance

RooNameReg& RooNameReg::instance()
{
  static RooNameReg instance;
  return instance;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a unique TNamed pointer for given C++ string

const TNamed* RooNameReg::constPtr(const char* inStr)
{
  // Handle null pointer case explicitly
  if (inStr==0) return 0 ;

  // See if name is already registered ;
  auto elm = _map.find(inStr) ;
  if (elm != _map.end()) return elm->second.get();

  // If not, register now
  auto t = make_unique<TNamed>(inStr,inStr);
  auto ret = t.get();
  _map.emplace(std::string(inStr), std::move(t));

  return ret;
}


////////////////////////////////////////////////////////////////////////////////
/// Return a unique TNamed pointer for given C++ string

const TNamed* RooNameReg::ptr(const char* stringPtr)
{
  if (stringPtr==0) return 0 ;
  return instance().constPtr(stringPtr) ;
}


////////////////////////////////////////////////////////////////////////////////
/// If the name is already known, return its TNamed pointer. Otherwise return 0 (don't register the name).

const TNamed* RooNameReg::known(const char* inStr)
{
  // Handle null pointer case explicitly
  if (inStr==0) return 0 ;
  RooNameReg& reg = instance();
  const auto elm = reg._map.find(inStr);
  return elm != reg._map.end() ? elm->second.get() : nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// The renaming counter has to be incremented every time a RooAbsArg is
/// renamed. This is a protected function, and only the friend class RooAbsArg
/// should call it when it gets renamed.

void RooNameReg::incrementRenameCounter() {
  ++instance()._renameCounter;
}


////////////////////////////////////////////////////////////////////////////////
// Return a reference to a counter that keeps track how often a RooAbsArg was
/// renamed in this RooFit process.

const std::size_t& RooNameReg::renameCounter() {
  return instance()._renameCounter;
}
