/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsString.cc,v 1.4 2001/05/03 02:15:54 verkerke Exp $
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
#include "TTree.h"
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



RooAbsString::RooAbsString(const RooAbsString& other, const char* name) : 
  RooAbsArg(other, name)
{
  strcpy(_value,other._value) ;
}



RooAbsString::~RooAbsString()
{
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
  return kFALSE ;
} 

void RooAbsString::writeToStream(ostream& os, Bool_t compact) const
{
  //Write object contents to stream (dummy for now)
}


void RooAbsString::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  //Print object contents
  os << indent << "RooAbsString: " << GetName() << " = " << getVal();
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



void RooAbsString::copyCache(const RooAbsArg* source) 
{
  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  RooAbsString* other = dynamic_cast<RooAbsString*>(const_cast<RooAbsArg*>(source)) ;
  assert(other) ;

  strcpy(_value,other->_value) ;
  setValueDirty(kTRUE) ;
}



void RooAbsString::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    t.SetBranchAddress(GetName(),&_value) ;
  } else {
    TString format(GetName());
    format.Append("/C");
    t.Branch(GetName(), &_value, (const Text_t*)format, bufSize);
  }
}
 
