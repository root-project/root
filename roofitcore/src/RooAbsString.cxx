/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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


RooAbsString::RooAbsString(const RooAbsString& other) : 
  RooAbsArg(other)
{
  strcpy(_value,other._value) ;
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}


RooAbsString::~RooAbsString()
{
}


RooAbsArg& RooAbsString::operator=(RooAbsArg& aother)
{
  RooAbsArg::operator=(aother) ;

  RooAbsString& other=(RooAbsString&)aother ;
  strcpy(_value,other._value) ;

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
  return *this ;
}


TString RooAbsString::getVal() 
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    setValueDirty(false) ;
    strcpy(_value,traceEval()) ;
  } 
  
  return TString(_value) ;
}



Bool_t RooAbsString::operator==(TString value) 
{
  return (getVal()==value) ;
}



Bool_t RooAbsString::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooAbsString::writeToStream(ostream& os, Bool_t compact)
{
  //Write object contents to stream (dummy for now)
}


void RooAbsString::printToStream(ostream& os, PrintOption opt) 
{
  //Print object contents
  os << "RooAbsString: " << GetName() << " = " << getVal();
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
}



Bool_t RooAbsString::isValid() {
  return isValid(getVal()) ;
}


Bool_t RooAbsString::isValid(TString value) {

  // Protect against string overflows
  if (value.Length()>1023) return kFALSE ;

  return kTRUE ;
}



TString RooAbsString::traceEval()
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




