/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.cc,v 1.6 2001/03/29 22:37:39 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include <stdlib.h>
#include "TString.h"
#include "TH1.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/Roo1DTable.hh"

ClassImp(RooAbsCategory) 
;


RooAbsCategory::RooAbsCategory(const char *name, const char *title) : 
  RooAbsArg(name,title)
{
  setValueDirty(kTRUE) ;  
  setShapeDirty(kTRUE) ;  
}



RooAbsCategory::RooAbsCategory(const char* name, const RooAbsCategory& other) :
  RooAbsArg(name, other), _value(other._value) 
{
  initCopy(other) ;
}



RooAbsCategory::RooAbsCategory(const RooAbsCategory& other) : 
  RooAbsArg(other), _value(other._value)
{
  initCopy(other) ;
}



void RooAbsCategory::initCopy(const RooAbsCategory& other)
{
  TIterator* iter=other._types.MakeIterator() ;
  TObject* obj ;
  while (obj=iter->Next()) {
    _types.Add(obj->Clone()) ;
  }
  delete iter ;

  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}



RooAbsCategory::~RooAbsCategory()
{
  // We own the contents of _types 
  _types.Delete() ;
}


RooAbsCategory& RooAbsCategory::operator=(RooAbsCategory& other)
{
  RooAbsArg::operator=(other) ;

  if (!lookupType(other._value)) {
    cout << "RooAbsCategory::operator=(" << GetName() << "): index " 
	 << other._value.getVal() << " not defined for this category" << endl ;
  } else {
    _value = other._value ;
    setValueDirty(kTRUE) ;
  }
  return *this ;
}

RooAbsArg& RooAbsCategory::operator=(RooAbsArg& aother)
{
  return operator=((RooAbsCategory&)aother) ;
}


TIterator* RooAbsCategory::typeIterator() const
{
  return _types.MakeIterator() ;
}


Int_t RooAbsCategory::getIndex() const
{
  if (isValueDirty()) {
    setValueDirty(false) ;
    _value = traceEval() ;
  } 

  return _value.getVal() ;
}


const char* RooAbsCategory::getLabel() const
{
  if (isValueDirty()) {
    setValueDirty(false) ;
    _value = traceEval() ;
  } 

  return _value.GetName() ;
}



Bool_t RooAbsCategory::operator==(Int_t index) const
{
  return (index==getIndex()) ;
}



Bool_t RooAbsCategory::operator==(const char* label) const
{
  return !TString(label).CompareTo(getLabel()) ;
}


Bool_t RooAbsCategory::isValidIndex(Int_t index) const
{
  return lookupType(index)?kTRUE:kFALSE ;
}


Bool_t RooAbsCategory::isValidLabel(const char* label) const
{
  return lookupType(label)?kTRUE:kFALSE ;
}


RooCatType RooAbsCategory::traceEval() const
{
  RooCatType value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


Int_t RooAbsCategory::getOrdinalIndex() const
{
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    if (*(RooCatType*)_types.At(i) == _value) return i ;
  }
  // This should never happen
  cout << "RooAbsCategory::getOrdinalIndex(" << GetName() 
       << "): Internal error: current type is illegal" << endl ;
  return -1 ;
}



Bool_t RooAbsCategory::setOrdinalIndex(Int_t newIndex) 
{
  if (newIndex<0 || newIndex>=_types.GetEntries()) {
    cout << "RooAbsCategory::setOrdinalIndex(" << GetName() 
	 << "): ordinal index out of range: " << newIndex 
	 << " (0," << _types.GetEntries() << ")" << endl ;
    return kTRUE ;
  }
  _value = *(RooCatType*)_types.At(newIndex) ;
  return kFALSE ;
}



Bool_t RooAbsCategory::defineType(Int_t index, const char* label) 
{
  if (isValidIndex(index)) {
    cout << "RooAbsCategory::defineType(" << GetName() << "): index " 
	 << index << " already assigned" << endl ;
    return kTRUE ;
  }

  if (isValidLabel(label)) {
    cout << "RooAbsCategory::defineType(" << GetName() << "): label " 
	 << label << " already assigned" << endl ;
    return kTRUE ;
  }

  
  Bool_t first = _types.GetEntries()?kFALSE:kTRUE ;
  _types.Add(new RooCatType(label,index)) ;

  if (first) _value = RooCatType(label,index) ;
  setShapeDirty(kTRUE) ;

  return kFALSE ;
}



const RooCatType* RooAbsCategory::lookupType(Int_t index, Bool_t printError) const
{
  RooCatType* type ;
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    type = (RooCatType*)_types.At(i) ;
    if (type->getVal()==index) return type ;
  }  

  if (printError) {
    cout << "RooAbsCategory::lookupType(" << GetName() 
	 << "): type " << index << " undefined" << endl ;
  }
  return 0 ;
}



const RooCatType* RooAbsCategory::lookupType(const char* label, Bool_t printError) const 
{
  char *endptr(0) ;
  Int_t val = strtol(label,&endptr,10) ;
  if (endptr-label==strlen(label)) {
    return lookupType(val,printError) ;
  }

  RooCatType* type ;
  for (int i=0 ; i<_types.GetEntries() ; i++) {
    type = (RooCatType*)_types.At(i) ;
    if (!TString(type->GetName()).CompareTo(label)) return type ;
  }  

  if (printError) {
    cout << "RooAbsCategory::lookupType(" << GetName() 
	 << "): type " << label << " undefined" << endl ;
  }
  return 0 ;
}


Bool_t RooAbsCategory::isValid() const
{
  return isValid(_value) ;
}


Bool_t RooAbsCategory::isValid(RooCatType value)  const
{
  return isValidIndex(value.getVal()) ;
}


Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  return new Roo1DTable(GetName(),label,*this) ;
}


Bool_t RooAbsCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooAbsCategory::writeToStream(ostream& os, Bool_t compact) const
{
  //Write object contents to stream (dummy for now)
}

void RooAbsCategory::printToStream(ostream& os, PrintOption opt) const
{
  if (_types.GetEntries()==0) {
    os << "RooAbsCategory: " << GetName() << " has no types defined" << endl ;
    return ;
  }

  //Print object contents
  if (opt==Shape) {
    // list all of the types defined by this object
    os << "RooAbsCategory: " << GetName() << ": \"" << GetTitle()
       << "\" defines the following categories:"
       << endl;
    for (int i=0 ; i<_types.GetEntries() ; i++) {
      os << "  [" << i << "] \"" << ((RooCatType*)_types.At(i))->GetName() << "\" (code = "
	 << ((RooCatType*)_types.At(i))->getVal() << ")" << endl;      
    }
  } else {
    os << "RooAbsCategory: " << GetName() << " = " << getLabel() << " (" << getIndex() << ")" ;
    os << " : \"" << fTitle << "\"" ;

    printAttribList(os) ;
    os << endl ;
  }
}


