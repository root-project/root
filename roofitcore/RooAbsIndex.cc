/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsIndex.cc,v 1.2 2001/03/16 07:59:11 verkerke Exp $
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
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
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

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
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

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}


TIterator* RooAbsIndex::typeIterator()
{
  return _types.MakeIterator() ;
}


Int_t RooAbsIndex::getIndex()
{
  if (isValueDirty()) {
    setValueDirty(false) ;
    _value = traceEval() ;
  } 

  return _value.getVal() ;
}


const char* RooAbsIndex::getLabel()
{
  if (isValueDirty()) {
    setValueDirty(false) ;
    _value = traceEval() ;
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


RooCat RooAbsIndex::traceEval()
{
  RooCat value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
    cout << "RooAbsIndex::traceEval(" << GetName() << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


Int_t RooAbsIndex::getOrdinalIndex() 
{
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    if (*(RooCat*)_types.At(i) == _value) return i ;
  }
  // This should never happen
  cout << "RooAbsIndex::getOrdinalIndex(" << GetName() << "): Internal error: current type is illegal" << endl ;
  return -1 ;
}



Bool_t RooAbsIndex::setOrdinalIndex(Int_t newIndex) 
{
  if (newIndex<0 || newIndex>=_types.GetEntries()) {
    cout << "RooAbsIndex::setOrdinalIndex(" << GetName() << "): ordinal index out of range: " 
	 << newIndex << " (0," << _types.GetEntries() << ")" << endl ;
    return kTRUE ;
  }
  _value = *(RooCat*)_types.At(newIndex) ;
  return kFALSE ;
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



Bool_t RooAbsIndex::isValid()
{
  return isValid(_value) ;
}


Bool_t RooAbsIndex::isValid(RooCat value) 
{
  return isValidIndex(value.getVal()) ;
}


Bool_t RooAbsIndex::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooAbsIndex::writeToStream(ostream& os, Bool_t compact)
{
  //Write object contents to stream (dummy for now)
}

void RooAbsIndex::printToStream(ostream& os, PrintOption opt) 
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
	 << ((RooCat*)_types.At(i))->getVal() << ")" << endl;      
    }
  } else {
    os << "RooAbsIndex: " << GetName() << " = " << getLabel() << "(" << getIndex() << ")" ;
    os << " : \"" << fTitle << "\"" ;

    printAttribList(os) ;
    os << endl ;
  }
}


