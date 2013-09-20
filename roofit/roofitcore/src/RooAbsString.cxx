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
// RooAbsString is the common abstract base class for objects that represent a
// string value
// 
// Implementation of RooAbsString may be derived, there no interface
// is provided to modify the contents
// END_HTML
//
// 

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "TObjString.h"
#include "TH1.h"
#include "TTree.h"

#include "RooArgSet.h"
#include "RooAbsString.h"
#include "RooStringVar.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooAbsString) 
;


//_____________________________________________________________________________
RooAbsString::RooAbsString() : RooAbsArg(), _len(128) , _value(new char[128])
{
  // Default constructor
}


//_____________________________________________________________________________
RooAbsString::RooAbsString(const char *name, const char *title, Int_t bufLen) : 
  RooAbsArg(name,title), _len(bufLen), _value(new char[bufLen]) 
{
  // Constructor

  setValueDirty() ;
  setShapeDirty() ;
}



//_____________________________________________________________________________
RooAbsString::RooAbsString(const RooAbsString& other, const char* name) : 
  RooAbsArg(other, name), _len(other._len), _value(new char[other._len])
{
  // Copy constructor

  strlcpy(_value,other._value,_len) ;
}



//_____________________________________________________________________________
RooAbsString::~RooAbsString()
{
  // Destructor

  delete[] _value ;
}



//_____________________________________________________________________________
const char* RooAbsString::getVal() const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.

  if (isValueDirty()) {
    clearValueDirty() ;
    strlcpy(_value,traceEval(),_len) ;
  } 
  
  return _value ;
}



//_____________________________________________________________________________
Bool_t RooAbsString::operator==(const char* value) const
{
  // Equality operator comparing with a TString

  return !TString(getVal()).CompareTo(value) ;
}


//_____________________________________________________________________________
Bool_t RooAbsString::isIdentical(const RooAbsArg& other, Bool_t assumeSameType)  
{
  if (!assumeSameType) {
    const RooAbsString* otherString = dynamic_cast<const RooAbsString*>(&other) ;
    return otherString ? operator==(otherString->getVal()) : kFALSE ;
  } else {
    return !TString(getVal()).CompareTo(((RooAbsString&)other).getVal()) ; ;
  }
}



//_____________________________________________________________________________
Bool_t RooAbsString::operator==(const RooAbsArg& other) 
{
  // Equality operator comparing to another RooAbsArg

  const RooAbsString* otherString = dynamic_cast<const RooAbsString*>(&other) ;
  return otherString ? operator==(otherString->getVal()) : kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooAbsString::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  //Read object contents from stream (dummy for now)
  return kFALSE ;
} 



//_____________________________________________________________________________
void RooAbsString::writeToStream(ostream& /*os*/, Bool_t /*compact*/) const
{
  //Write object contents to stream (dummy for now)
}



//_____________________________________________________________________________
void RooAbsString::printValue(ostream& os) const
{
  // Print value
  os << getVal() ;
}



//_____________________________________________________________________________
Bool_t RooAbsString::isValid() const 
{
  // Check if current value is valid
  return isValidString(getVal()) ;
}



//_____________________________________________________________________________
Bool_t RooAbsString::isValidString(const char* value, Bool_t /*printError*/) const 
{
  // Check if given string value is valid

  // Protect against string overflows
  if (TString(value).Length()>_len) return kFALSE ;

  return kTRUE ;
}


//_____________________________________________________________________________
Bool_t RooAbsString::traceEvalHook(const char* /*value*/) const 
{ 
  // Hook function for trace evaluation
  return kFALSE ; 
}



//_____________________________________________________________________________
const char* RooAbsString::traceEval() const
{
  // Calculate current value of object, with error tracing wrapper

  const char* value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValidString(value)) {
    cxcoutD(Tracing) << "RooAbsString::traceEval(" << GetName() << "): new output too long (>" << _len << " chars): " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}



//_____________________________________________________________________________
void RooAbsString::syncCache(const RooArgSet*) 
{ 
  // Forcibly bring internal cache up-to-date
  getVal() ; 
}



//_____________________________________________________________________________
void RooAbsString::copyCache(const RooAbsArg* source, Bool_t /*valueOnly*/, Bool_t setValDirty) 
{
  // Copy cache of another RooAbsArg to our cache
  //
  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  RooAbsString* other = dynamic_cast<RooAbsString*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  strlcpy(_value,other->_value,_len) ;
  if (setValDirty) {
    setValueDirty() ;
  }
}



//_____________________________________________________________________________
void RooAbsString::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  TBranch* branch ;
  if ((branch = t.GetBranch(GetName()))) {
    t.SetBranchAddress(GetName(),_value) ;
    if (branch->GetCompressionLevel()<0) {
      cxcoutD(DataHandling) << "RooAbsString::attachToTree(" << GetName() << ") Fixing compression level of branch " << GetName() << endl ;
      branch->SetCompressionLevel(1) ;
    }
  } else {
    TString format(GetName());
    format.Append("/C");
    branch = t.Branch(GetName(), _value, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(1) ;
  }
}
 


//_____________________________________________________________________________
void RooAbsString::fillTreeBranch(TTree& t) 
{
  // Fill tree branch associated with this object

  // First determine if branch is taken
  TBranch* branch = t.GetBranch(GetName()) ;
  if (!branch) { 
    coutE(DataHandling) << "RooAbsString::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }
  branch->Fill() ;  
}



//_____________________________________________________________________________
void RooAbsString::setTreeBranchStatus(TTree& t, Bool_t active) 
{
  // (De)Activate associated tree branch

  TBranch* branch = t.GetBranch(GetName()) ;
  if (branch) { 
    t.SetBranchStatus(GetName(),active?1:0) ;
  }
}



//_____________________________________________________________________________
RooAbsArg *RooAbsString::createFundamental(const char* newname) const 
{
  // Create a RooStringVar fundamental object with our properties.

  RooStringVar *fund= new RooStringVar(newname?newname:GetName(),GetTitle(),"") ; 
  return fund;
}
