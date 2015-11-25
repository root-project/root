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
\file RooStringVar.cxx
\class RooStringVar
\ingroup Roofitcore

RooStringVar implements a string values RooAbsArg
**/

#include "RooFit.h"
#include "Riostream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooStringVar.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"



using namespace std;

ClassImp(RooStringVar)


              

////////////////////////////////////////////////////////////////////////////////
/// Constructor with initial value and internal buffer size

RooStringVar::RooStringVar(const char *name, const char *title, const char* value, Int_t size) :
  RooAbsString(name, title, size)
{
  if(!isValidString(value)) {
    coutW(InputArguments) << "RooStringVar::RooStringVar(" << GetName() 
	 << "): initial contents too long and ignored" << endl ;
  } else {
    strlcpy(_value,value,_len) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}  



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooStringVar::RooStringVar(const RooStringVar& other, const char* name) :
  RooAbsString(other, name)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooStringVar::~RooStringVar() 
{
}



////////////////////////////////////////////////////////////////////////////////

RooStringVar::operator TString() 
{
  // Cast operator to TString
  return TString(_value) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set value to given TString

void RooStringVar::setVal(const char* value) 
{
  if (!isValidString(value)) {    
    coutW(InputArguments) << "RooStringVar::setVal(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    if (value) {
      strlcpy(_value,value,_len) ;
    } else {
      _value[0] = 0 ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Set value to given TString

RooAbsArg& RooStringVar::operator=(const char* newValue) 
{
  if (!isValidString(newValue)) {
    coutW(InputArguments) << "RooStringVar::operator=(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    if (newValue) {
      strlcpy(_value,newValue,_len) ;
    } else {
      _value[0] = 0 ;
    }
  }

  return *this ;
}



////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream

Bool_t RooStringVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  TString token,errorPrefix("RooStringVar::readFromStream(") ;
  errorPrefix.Append(GetName()) ;
  errorPrefix.Append(")") ;
  RooStreamParser parser(is,errorPrefix) ;

  TString newValue ;
  Bool_t ret(kFALSE) ;

  if (compact) {
    parser.readString(newValue,kTRUE) ;
  } else {
    newValue = parser.readLine() ;
  }
  
  if (!isValidString(newValue)) {
    if (verbose) 
      coutW(InputArguments) << "RooStringVar::readFromStream(" << GetName() 
			    << "): new string too long and ignored" << endl ;
  } else {
    strlcpy(_value,newValue,_len) ;
  }

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Write object contents to given stream

void RooStringVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  os << getVal() ;
}


