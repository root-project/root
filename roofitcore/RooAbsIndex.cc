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
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsIndex.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooAbsIndex) 
;


RooAbsIndex::RooAbsIndex(const char *name, const char *title) : 
  RooAbsArg(name,title)
{
  _typeIter = _types.MakeIterator() ;
  setDirty(kTRUE) ;  
}


RooAbsIndex::RooAbsIndex(const RooAbsIndex& other) :
  RooAbsArg(other), _value(other._value) 
{
  TIterator* iter=other._types.MakeIterator() ;
  TObject* obj ;
  while (obj=iter->Next()) {
    _types.Add(obj) ;
  }
  delete iter ;

  setDirty(kTRUE) ;
}


RooAbsIndex::~RooAbsIndex()
{
  // We own the contents of _types 
  _types.Delete() ;
}


RooAbsArg& RooAbsIndex::operator=(RooAbsArg& aother)
{
  RooAbsArg::operator=(aother) ;
  RooAbsIndex& other=(RooAbsIndex&)aother ;

  _value = other._value ;

  _types.Delete() ;
  TIterator* iter=other._types.MakeIterator() ;
  TObject* obj ;
  while (obj=iter->Next()) {
    _types.Add(obj) ;
  }
  delete iter ;

  setDirty(kTRUE) ;
}


TIterator* RooAbsIndex::typeIterator()
{
  return _types.MakeIterator() ;
}


Int_t RooAbsIndex::getIndex()
{
  if (isDirty()) {
    setDirty(false) ;
    _value = Evaluate() ;
  } 

  return _value.GetVar() ;
}


const char* RooAbsIndex::getLabel()
{
  if (isDirty()) {
    setDirty(false) ;
    _value = Evaluate() ;
  } 

  return _value.GetName() ;
}



Bool_t RooAbsIndex::isValidIndex(Int_t index) 
{
  TIterator* tIter = typeIterator() ;
  RooCat* type ;
  while(type=(RooCat*)tIter->Next()) {
    if ((Int_t)(*type) == index) return kTRUE ;
  }
  return kFALSE ;
}


Bool_t RooAbsIndex::isValidLabel(char* label)
{
  TIterator* tIter = typeIterator() ;
  RooCat* type ;
  while(type=(RooCat*)tIter->Next()) {
    if (!TString(type->GetName()).CompareTo(label)) return kTRUE ;
  }
  return kFALSE ;
}


Bool_t RooAbsIndex::isValid() 
{
  return isValidIndex(_value.GetVar()) ;
}



Bool_t RooAbsIndex::defineType(Int_t index, char* label) 
{
  if (isValidIndex(index)) {
    cout << "RooAbsIndex::defineType(" << GetName() << "): index " << index << " already assigned" << endl ;
    return kTRUE ;
  }

  if (isValidLabel(label)) {
    cout << "RooAbsIdex::defineType(" << GetName() << "): label " << label << " already assigned" << endl ;
    return kTRUE ;
  }

  Bool_t first = _types.GetEntries()?kFALSE:kTRUE ;
  _types.Add(new RooCat(label,index)) ;

  return kFALSE ;
}



Bool_t RooAbsIndex::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooAbsIndex::writeToStream(ostream& os, Bool_t compact)
{
  //Write object contents to stream (dummy for now)
}

void RooAbsIndex::PrintToStream(ostream& os, Option_t* opt= 0) 
{
  if (_types.GetEntries()==0) {
    os << "RooAbsIndex: " << GetName() << " has no types defined" << endl ;
    return ;
  }

  //Print object contents
  if (TString(opt).Contains("t")) {
    // list all of the types defined by this object
    os << "RooAbsIndex: " << GetName() << ": \"" << GetTitle()
       << "\" defines the following categories:"
       << endl;
    for (int i=0 ; i<_types.GetEntries() ; i++) {
      os << "  [" << i << "] \"" << ((RooCat*)_types.At(i))->GetName() << "\" (code = "
	 << ((RooCat*)_types.At(i))->GetVar() << ")" << endl;      
    }
  } else {
    os << "RooAbsIndex: " << GetName() << " = " << getLabel() << "(" << getIndex() << ")" ;
    os << " : \"" << fTitle << "\"" ;

    printAttribList(os) ;
    os << endl ;
  }
}


