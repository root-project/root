/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsString.cc,v 1.16 2002/01/10 00:09:00 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooAbsString is the common abstract base class for objects that represent a
// string value
// 
// Implementation of RooAbsString may be derived, there no interface
// is provided to modify the contents
// 

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "TTree.h"

#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsString.hh"
#include "RooFitCore/RooStringVar.hh"

ClassImp(RooAbsString) 
;


RooAbsString::RooAbsString(const char *name, const char *title, Int_t bufLen) : 
  RooAbsArg(name,title), _value(new char[bufLen]), _len(bufLen)
{
  // Constructor
  setValueDirty() ;
  setShapeDirty() ;
}



RooAbsString::RooAbsString(const RooAbsString& other, const char* name) : 
  RooAbsArg(other, name), _value(new char[other._len]), _len(other._len)
{
  // Copy constructor
  strcpy(_value,other._value) ;
}



RooAbsString::~RooAbsString()
{
  delete[] _value ;
  // Destructor
}


TString RooAbsString::getVal() const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty()) {
    clearValueDirty() ;
    strcpy(_value,traceEval()) ;
  } 
  
  return TString(_value) ;
}



Bool_t RooAbsString::operator==(TString value) const
{
  // Equality operator comparing with a TString
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
  os << "RooAbsString: " << GetName() << " = " << getVal();
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
}



Bool_t RooAbsString::isValid() const 
{
  // Check if current value is valid
  return isValidString(getVal()) ;
}


Bool_t RooAbsString::isValidString(TString value, Bool_t printError) const 
{
  // Check if given value is valid

  // Protect against string overflows
  if (value.Length()>_len) return kFALSE ;

  return kTRUE ;
}



TString RooAbsString::traceEval() const
{
  // Calculate current value of object, with error tracing wrapper
  TString value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValidString(value)) {
    cout << "RooAbsString::traceEval(" << GetName() << "): new output too long (>" << _len << " chars): " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}



void RooAbsString::copyCache(const RooAbsArg* source) 
{
  // Copy cache of another RooAbsArg to our cache

  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  RooAbsString* other = dynamic_cast<RooAbsString*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  strcpy(_value,other->_value) ;
  setValueDirty() ;
}



void RooAbsString::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  TBranch* branch ;
  if (branch = t.GetBranch(GetName())) {
    t.SetBranchAddress(GetName(),_value) ;
    if (branch->GetCompressionLevel()<0) {
      cout << "RooAbsString::attachToTree(" << GetName() << ") Fixing compression level of branch " << GetName() << endl ;
      branch->SetCompressionLevel(1) ;
    }
  } else {
    TString format(GetName());
    format.Append("/C");
    branch = t.Branch(GetName(), _value, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(1) ;
  }
}
 


void RooAbsString::fillTreeBranch(TTree& t) 
{
  // First determine if branch is taken
  TBranch* branch = t.GetBranch(GetName()) ;
  if (!branch) { 
    cout << "RooAbsString::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }
  branch->Fill() ;  
}



RooAbsArg *RooAbsString::createFundamental(const char* newname) const {
  // Create a RooStringVar fundamental object with our properties.

  RooStringVar *fund= new RooStringVar(newname?newname:GetName(),GetTitle(),"") ; 
  return fund;
}


Int_t RooAbsString::getPlotBin() const 
{
  return 0 ;
}


RooAbsBinIter* RooAbsString::createPlotBinIterator() const 
{
  return 0 ;
}
