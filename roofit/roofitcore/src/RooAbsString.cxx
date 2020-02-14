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
\file RooAbsString.cxx
\class RooAbsString
\ingroup Roofitcore

RooAbsString is the common abstract base class for objects that represent a
string value

Implementation of RooAbsString may be derived, there no interface
is provided to modify the contents
**/
//

#include "RooFit.h"

#include "Compression.h"
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

ClassImp(RooAbsString);
;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsString::RooAbsString() : RooAbsArg(), _len(128) , _value(new char[128])
{
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsString::RooAbsString(const char *name, const char *title, Int_t bufLen) :
  RooAbsArg(name,title), _len(bufLen), _value(new char[bufLen])
{
  setValueDirty() ;
  setShapeDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsString::RooAbsString(const RooAbsString& other, const char* name) :
  RooAbsArg(other, name), _len(other._len), _value(new char[other._len])
{
  strlcpy(_value,other._value,_len) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsString::~RooAbsString()
{
  delete[] _value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return value of object. Calculated if dirty, otherwise cached value is returned.

const char* RooAbsString::getVal() const
{
  if (isValueDirty()) {
    clearValueDirty() ;
    strlcpy(_value,traceEval(),_len) ;
  }

  return _value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator comparing with a TString

Bool_t RooAbsString::operator==(const char* value) const
{
  return !TString(getVal()).CompareTo(value) ;
}


////////////////////////////////////////////////////////////////////////////////

Bool_t RooAbsString::isIdentical(const RooAbsArg& other, Bool_t assumeSameType) const
{
  if (!assumeSameType) {
    const RooAbsString* otherString = dynamic_cast<const RooAbsString*>(&other) ;
    return otherString ? operator==(otherString->getVal()) : kFALSE ;
  } else {
    return !TString(getVal()).CompareTo(((RooAbsString&)other).getVal()) ; ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator comparing to another RooAbsArg

Bool_t RooAbsString::operator==(const RooAbsArg& other) const
{
  const RooAbsString* otherString = dynamic_cast<const RooAbsString*>(&other) ;
  return otherString ? operator==(otherString->getVal()) : kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
///Read object contents from stream (dummy for now)

Bool_t RooAbsString::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/)
{
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
///Write object contents to stream (dummy for now)

void RooAbsString::writeToStream(ostream& /*os*/, Bool_t /*compact*/) const
{
}



////////////////////////////////////////////////////////////////////////////////
/// Print value

void RooAbsString::printValue(ostream& os) const
{
  os << getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if current value is valid

Bool_t RooAbsString::isValid() const
{
  return isValidString(getVal()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if given string value is valid

Bool_t RooAbsString::isValidString(const char* value, Bool_t /*printError*/) const
{
  // Protect against string overflows
  if (TString(value).Length()>_len) return kFALSE ;

  return kTRUE ;
}


////////////////////////////////////////////////////////////////////////////////
/// Hook function for trace evaluation

Bool_t RooAbsString::traceEvalHook(const char* /*value*/) const
{
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of object, with error tracing wrapper

TString RooAbsString::traceEval() const
{
  TString value = evaluate() ;

  //Standard tracing code goes here
  if (!isValidString(value)) {
    cxcoutD(Tracing) << "RooAbsString::traceEval(" << GetName() << "): new output too long (>" << _len << " chars): " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Forcibly bring internal cache up-to-date

void RooAbsString::syncCache(const RooArgSet*)
{
  getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy cache of another RooAbsArg to our cache
///
/// Warning: This function copies the cached values of source,
///          it is the callers responsibility to make sure the cache is clean

void RooAbsString::copyCache(const RooAbsArg* source, Bool_t /*valueOnly*/, Bool_t setValDirty)
{
  RooAbsString* other = dynamic_cast<RooAbsString*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  strlcpy(_value,other->_value,_len) ;
  if (setValDirty) {
    setValueDirty() ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Attach object to a branch of given TTree

void RooAbsString::attachToTree(TTree& t, Int_t bufSize)
{
  // First determine if branch is taken
  TBranch* branch ;
  if ((branch = t.GetBranch(GetName()))) {
    t.SetBranchAddress(GetName(),_value) ;
    if (branch->GetCompressionLevel()<0) {
      cxcoutD(DataHandling) << "RooAbsString::attachToTree(" << GetName() << ") Fixing compression level of branch " << GetName() << endl ;
      branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
    }
  } else {
    TString format(GetName());
    format.Append("/C");
    branch = t.Branch(GetName(), _value, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Fill tree branch associated with this object

void RooAbsString::fillTreeBranch(TTree& t)
{
  // First determine if branch is taken
  TBranch* branch = t.GetBranch(GetName()) ;
  if (!branch) {
    coutE(DataHandling) << "RooAbsString::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }
  branch->Fill() ;
}



////////////////////////////////////////////////////////////////////////////////
/// (De)Activate associated tree branch

void RooAbsString::setTreeBranchStatus(TTree& t, Bool_t active)
{
  TBranch* branch = t.GetBranch(GetName()) ;
  if (branch) {
    t.SetBranchStatus(GetName(),active?1:0) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooStringVar fundamental object with our properties.

RooAbsArg *RooAbsString::createFundamental(const char* newname) const
{
  RooStringVar *fund= new RooStringVar(newname?newname:GetName(),GetTitle(),"") ;
  return fund;
}
