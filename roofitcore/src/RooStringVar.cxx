/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStringVar.cc,v 1.5 2001/05/03 02:15:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

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
  if(!isValid(value)) {
    cout << "RooStringVar::RooStringVar(" << GetName() 
	 << "): initial contents too long and ignored" << endl ;
  } else {
    strcpy(_value,value) ;
  }

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}  


RooStringVar::RooStringVar(const RooStringVar& other, const char* name) :
  RooAbsString(other, name)
{
}


RooStringVar::~RooStringVar() 
{
}


RooStringVar::operator TString() {
  return this->getVal();
}


void RooStringVar::setVal(TString value) {
  if (!isValid(value)) {    
    cout << "RooStringVar::setVal(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,value) ;
  }
}



TString RooStringVar::operator=(TString newValue) 
{
  if (!isValid(newValue)) {
    cout << "RooStringVar::operator=(" << GetName() << "): new string too long and ignored" << endl ;
  } else {
    strcpy(_value,newValue) ;
  }

  return newValue ;
}



Bool_t RooStringVar::isValid() const
{
  return isValid(getVal()) ;
}


Bool_t RooStringVar::isValid(TString value, Bool_t verbose) const {
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
  if (!isValid(newValue)) {
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


