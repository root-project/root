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
// RooMapCatEntry is an auxilary class for RooMappedCategory and defines a 
// a mapping. A mapping consists of a wildcard regular expression, that
// can be matched against the input category state label and an output
// category state (RooCatType), which should be assign if the match is successfull.

#include "RooFit.h"

#include "RooMapCatEntry.h"
#include "TString.h"

using namespace std;

ClassImp(RooMapCatEntry);
;

RooMapCatEntry::RooMapCatEntry(const char* exp, const RooCatType* cat) : 
  TNamed(exp,mangle(exp).Data()), _regexp(mangle(exp),kTRUE), _cat(*cat) 
{
}


RooMapCatEntry::RooMapCatEntry(const RooMapCatEntry& other) : 
  TNamed(other), _regexp(other.GetTitle(),kTRUE), _cat(other._cat) 
{
}


Bool_t RooMapCatEntry::match(const char* testPattern) const 
{
  return (TString(testPattern).Index(_regexp)>=0) ;
}



TString RooMapCatEntry::mangle(const char* exp) const
{
  // Mangle name : escape regexp character '+'
  TString t ;
  const char *c = exp ;
  while(*c) {
    if (*c=='+') t.Append('\\') ;
    t.Append(*c) ;
    c++ ;
  }
  return t ;
}
