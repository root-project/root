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
// RooStringVar implements a string values RooAbsArg
// END_HTML
//

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


              

//_____________________________________________________________________________
RooStringVar::RooStringVar(const char *name, const char *title, const char* value, Int_t size) :
  RooAbsString(name, title, size)
{
  // Constructor with initial value and internal buffer size

  if(!isValidString(value)) {
    coutW(InputArguments) << "RooStringVar::RooStringVar(" << GetName() 
	 << "): initial contents too long and ignored" << endl ;
  } else {
    strlcpy(_value,value,_len) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}  



//_____________________________________________________________________________
RooStringVar::RooStringVar(const RooStringVar& other, const char* name) :
  RooAbsString(other, name)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooStringVar::~RooStringVar() 
{
  // Destructor
}



//_____________________________________________________________________________
RooStringVar::operator TString() 
{

  // Cast operator to TString
  return TString(_value) ;
}



//_____________________________________________________________________________
void RooStringVar::setVal(const char* value) 
{
  // Set value to given TString

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



//_____________________________________________________________________________
RooAbsArg& RooStringVar::operator=(const char* newValue) 
{
  // Set value to given TString

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



//_____________________________________________________________________________
Bool_t RooStringVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

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


//_____________________________________________________________________________
void RooStringVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  // Write object contents to given stream

  os << getVal() ;
}


