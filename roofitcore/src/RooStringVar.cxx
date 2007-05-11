/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooStringVar.cc,v 1.26 2005/06/20 15:45:14 wverkerke Exp $
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
// RooStringVar represents a fundamental string valued object.

#include "RooFit.h"

#include <math.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooStringVar.h"
#include "RooStreamParser.h"

ClassImp(RooStringVar)


              
RooStringVar::RooStringVar(const char *name, const char *title, const char* value, Int_t size) :
  RooAbsString(name, title, size)
{
  // Constructor with initial value
  if(!isValidString(value)) {
    cout << "RooStringVar::RooStringVar(" << GetName() 
	 << "): initial contents too long and ignored" << endl ;
  } else {
    strcpy(_value,value) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}  


RooStringVar::RooStringVar(const RooStringVar& other, const char* name) :
  RooAbsString(other, name)
{
  // Copy constructor
}


RooStringVar::~RooStringVar() 
{
  // Destructor
}


RooStringVar::operator TString() {
  // Cast operator to TString
  return TString(_value) ;
}


void RooStringVar::setVal(const char* value) {
  // Set value to given TString
  if (!isValidString(value)) {    
    cout << "RooStringVar::setVal(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    if (value) {
      strcpy(_value,value) ;
    } else {
      _value[0] = 0 ;
    }
  }
}



RooAbsArg& RooStringVar::operator=(const char* newValue) 
{
  // Set value to given TString
  if (!isValidString(newValue)) {
    cout << "RooStringVar::operator=(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    if (newValue) {
      strcpy(_value,newValue) ;
    } else {
      _value[0] = 0 ;
    }
  }

  return *this ;
}



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
      cout << "RooStringVar::readFromStream(" << GetName() 
	   << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,newValue) ;
  }

  return ret ;
}

void RooStringVar::writeToStream(ostream& os, Bool_t /*compact*/) const
{
  // Write object contents to given stream
  os << getVal() ;
}


