/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.cc,v 1.9 2001/04/09 04:29:34 verkerke Exp $
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

RooAbsCategory& RooAbsCategory::operator=(const RooAbsCategory& other)
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

RooAbsArg& RooAbsCategory::operator=(const RooAbsArg& aother)
{
  return operator=((const RooAbsCategory&)aother) ;
}

TIterator* RooAbsCategory::typeIterator() const
{
  return _types.MakeIterator() ;
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

Bool_t RooAbsCategory::defineType(const char* label, Int_t index) 
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

void RooAbsCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //  Standard : label and index
  //     Shape : defined types
  //   Verbose : default binning and print label

  RooAbsArg::printToStream(os,opt,indent);
  if(opt >= Standard) {
    os << indent << "--- RooAbsCategory ---" << endl;
    if (_types.GetEntries()==0) {
      os << indent << "  ** No values defined **" << endl;
      return;
    }
    os << indent << "  Value is \"" << getLabel() << "\" (" << getIndex() << ")" << endl;
    if(opt >= Shape) {
      os << indent << "  Has the following possible values:" << endl;
      Int_t n= _types.GetEntries();
      for (int i=0 ; i < n ; i++) {
	os << indent
	   << ((RooCatType*)_types.At(i))->GetName() << "\" ("
	   << ((RooCatType*)_types.At(i))->getVal() << ")" << endl;      
      }
    }
  }
}


