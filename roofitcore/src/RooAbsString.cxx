/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsString.cc,v 1.2 2001/03/29 01:59:09 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsString.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooAbsString) 
;


RooAbsString::RooAbsString(const char *name, const char *title) : 
  RooAbsArg(name,title)
{
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}



RooAbsString::RooAbsString(const char* name, const RooAbsString& other) : 
  RooAbsArg(name, other)
{
  initCopy(other) ;
}



RooAbsString::RooAbsString(const RooAbsString& other) : 
  RooAbsArg(other)
{
  initCopy(other) ;
}



void RooAbsString::initCopy(const RooAbsString& other)
{
  strcpy(_value,other._value) ;
}


RooAbsString::~RooAbsString()
{
}


RooAbsString& RooAbsString::operator=(const RooAbsString& other)
{
  RooAbsArg::operator=(other) ;

  strcpy(_value,other._value) ;
  setValueDirty(kTRUE) ;

  return *this ;
}


RooAbsArg& RooAbsString::operator=(const RooAbsArg& aother)
{
  return operator=((RooAbsString&)aother) ;
}


TString RooAbsString::getVal() const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    setValueDirty(false) ;
    strcpy(_value,traceEval()) ;
  } 
  
  return TString(_value) ;
}



Bool_t RooAbsString::operator==(TString value) const
{
  return (getVal()==value) ;
}



Bool_t RooAbsString::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooAbsString::writeToStream(ostream& os, Bool_t compact) const
{
  //Write object contents to stream (dummy for now)
}


void RooAbsString::printToStream(ostream& os, PrintOption opt) const
{
  //Print object contents
  os << "RooAbsString: " << GetName() << " = " << getVal();
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
}



Bool_t RooAbsString::isValid() const 
{
  return isValid(getVal()) ;
}


Bool_t RooAbsString::isValid(TString value) const 
{

  // Protect against string overflows
  if (value.Length()>1023) return kFALSE ;

  return kTRUE ;
}



TString RooAbsString::traceEval() const
{
  TString value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
    cout << "RooAbsString::traceEval(" << GetName() << "): new output too long (>1023 chars): " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}




