/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStringVar.cc,v 1.9 2001/07/31 05:54:22 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooStringVar represents a fundamental (non-derived) string value object.

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooStringVar)


              
RooStringVar::RooStringVar(const char *name, const char *title, const char* value) :
  RooAbsString(name, title)
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
  return this->getVal();
}


void RooStringVar::setVal(TString value) {
  // Set value to given TString
  if (!isValidString(value)) {    
    cout << "RooStringVar::setVal(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,value) ;
  }
}



TString RooStringVar::operator=(TString newValue) 
{
  // Set value to given TString
  if (!isValidString(newValue)) {
    cout << "RooStringVar::operator=(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,newValue) ;
  }

  return newValue ;
}




Bool_t RooStringVar::isValidString(TString value, Bool_t verbose) const {
  // Test if given TString is valid
  return kTRUE ;
}



Bool_t RooStringVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  TString token,errorPrefix("RooStringVar::readFromStream(") ;
  errorPrefix.Append(GetName()) ;
  errorPrefix.Append(")") ;
  RooStreamParser parser(is,errorPrefix) ;

  TString newValue ;
  Bool_t ret = parser.readString(newValue,kTRUE) ;
  if (!isValidString(newValue)) {
    if (verbose) 
      cout << "RooStringVar::readFromStreeam(" << GetName() 
	   << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,newValue) ;
  }
  return ret ;
}


void RooStringVar::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  os << getVal() ;
}


