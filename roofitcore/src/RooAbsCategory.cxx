/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.cc,v 1.23 2001/08/03 18:11:33 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooAbsCategory is the common abstract base class for objects that represent a
// discrete value with a finite number of states. Each state consist of a 
// label/index pair, which is stored in a RooCatType object.
// 
// Implementation of RooAbsCategory may be derived, there no interface
// is provided to modify the contents, nor a public interface to define states.

#include <iostream.h>
#include <stdlib.h>
#include "TString.h"
#include "TH1.h"
#include "TTree.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooCatBinIter.hh"

ClassImp(RooAbsCategory) 
;

RooAbsCategory::RooAbsCategory(const char *name, const char *title) : 
  RooAbsArg(name,title)
{
  // Constructor
  setValueDirty() ;  
  setShapeDirty() ;  
}

RooAbsCategory::RooAbsCategory(const RooAbsCategory& other,const char* name) :
  RooAbsArg(other,name), _value(other._value) 
{
  // Copy constructor
  TIterator* iter=other._types.MakeIterator() ;
  TObject* obj ;
  while (obj=iter->Next()) {
    //cout << "RooAbsCategory::RooAbsCategory(" << GetName() << "): cloning type " << ((RooAbsCategory*)obj)->GetName() << endl ;
    _types.Add(obj->Clone()) ;
  }
  delete iter ;

  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsCategory::~RooAbsCategory()
{
  // Destructor

  // We own the contents of _types 
  _types.Delete() ;
}


Int_t RooAbsCategory::getIndex() const
{
  // Return index number of current state 
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    clearValueDirty() ;
    clearShapeDirty() ;
  }

  return _value.getVal() ;
}


const char* RooAbsCategory::getLabel() const
{
  // Return label string of current state 
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    clearValueDirty() ;
    clearShapeDirty() ;
  }

  return _value.GetName() ;
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



TIterator* RooAbsCategory::typeIterator() const
{
  // Return iterator over all defined states
  return _types.MakeIterator() ;
}


Bool_t RooAbsCategory::operator==(Int_t index) const
{
  // Equality operator with a integer (compares with state index number)
  return (index==getIndex()) ;
}

Bool_t RooAbsCategory::operator==(const char* label) const
{
  // Equality operator with a string (compares with state label string)
  return !TString(label).CompareTo(getLabel()) ;
}

Bool_t RooAbsCategory::isValidIndex(Int_t index) const
{
  // Check if state with given index is defined
  return lookupType(index)?kTRUE:kFALSE ;
}

Bool_t RooAbsCategory::isValidLabel(const char* label) const
{
  // Check if state with given name is defined
  return lookupType(label)?kTRUE:kFALSE ;
}



const RooCatType* RooAbsCategory::defineType(const char* label)
{
  // Define a new state with given name. Index number is automatically assigned

  // Find lowest unused index
  Int_t index(-1) ;
  while(lookupType(++index,kFALSE)) ;
  
  // Assign this index to given label 
  return defineType(label,index) ;
}



const RooCatType* RooAbsCategory::defineType(const char* label, Int_t index) 
{
  // Define new state with given name and index number

  if (isValidIndex(index)) {
    cout << "RooAbsCategory::defineType(" << GetName() << "): index " 
	 << index << " already assigned" << endl ;
    return 0 ;
  }

  if (isValidLabel(label)) {
    cout << "RooAbsCategory::defineType(" << GetName() << "): label " 
	 << label << " already assigned or not allowed" << endl ;
    return 0 ;
  }

  Bool_t first = _types.GetEntries()?kFALSE:kTRUE ;
  RooCatType *newType = new RooCatType(label,index) ;
  _types.Add(newType) ;

  if (first) _value = RooCatType(label,index) ;
  setShapeDirty() ;

  return newType ;
}



void RooAbsCategory::clearTypes() 
{
  // Delete all currently defined states

  _types.Delete() ;
  _value = RooCatType("",0) ;
  setShapeDirty() ;
}



const RooCatType* RooAbsCategory::lookupType(const RooCatType &other, Bool_t printError) const 
{
  // Find our type that matches the specified type, or return 0 for no match.
  RooCatType* type ;
  Int_t n= _types.GetEntries();
  for (int i=0 ; i < n; i++) {
    type = (RooCatType*)_types.At(i) ;
    if((*type) == other) return type; // delegate comparison to RooCatType
  }

  if (printError) {
    cout << ClassName() << "::" << GetName() << ":lookupType: no match for ";
    other.printToStream(cout,OneLine);
  }
  return 0 ;
}

const RooCatType* RooAbsCategory::lookupType(Int_t index, Bool_t printError) const
{
  // Find our type corresponding to the specified index, or return 0 for no match.
  RooCatType* type ;
  Int_t n= _types.GetEntries();
  for (int i=0 ; i < n; i++) {
    type = (RooCatType*)_types.At(i) ;
    if((*type) == index) return type; // delegate comparison to RooCatType
  }
  if (printError) {
    cout << ClassName() << "::" << GetName() << ":lookupType: no match for index "
	 << index << endl;
  }
  return 0 ;
}

const RooCatType* RooAbsCategory::lookupType(const char* label, Bool_t printError) const 
{
  // Find our type corresponding to the specified label, or return 0 for no match.

  RooCatType* type ;
  Int_t n= _types.GetEntries();
  for (int i=0 ; i < n; i++) {
    type = (RooCatType*)_types.At(i) ;
    if((*type) == label) return type; // delegate comparison to RooCatType
  }

  // Try if label represents integer number
  char* endptr ;
  Int_t idx=strtol(label,&endptr,10)  ;
  if (endptr==label+strlen(label)) {
    for (int i=0 ; i < n; i++) {
    type = (RooCatType*)_types.At(i) ;
    if((*type) == idx) return type; // delegate comparison to RooCatType
    }
  }

  if (printError) {
    cout << ClassName() << "::" << GetName() << ":lookupType: no match for label "
	 << label << endl;
  }
  return 0 ;
}

Bool_t RooAbsCategory::isValid() const
{
  // Check if current value is a valid state
  return isValid(_value) ;
}

Bool_t RooAbsCategory::isValid(RooCatType value)  const
{
  // check if given state is defined for this object
  return isValidIndex(value.getVal()) ;
}

Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  // Create a table matching the shape of this category
  return new Roo1DTable(GetName(),label,*this) ;
}

Bool_t RooAbsCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
  return kFALSE ;
} 

void RooAbsCategory::writeToStream(ostream& os, Bool_t compact) const
{
  //Write object contents to stream 
  if (compact) {
    os << getLabel() ;
  } else {
    os << getLabel() ;
  }
}

void RooAbsCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //     Shape : label, index, defined types

  RooAbsArg::printToStream(os,opt,indent);
  if(opt >= Shape) {
    os << indent << "--- RooAbsCategory ---" << endl;
    if (_types.GetEntries()==0) {
      os << indent << "  ** No values defined **" << endl;
      return;
    }
    os << indent << "  Value is \"" << getLabel() << "\" (" << getIndex() << ")" << endl;
    os << indent << "  Has the following possible values:" << endl;
    indent.Append("    ");
    opt= lessVerbose(opt);
    RooCatType *type;
    Int_t n= _types.GetEntries();
    for (int i=0 ; i < n ; i++) {
      type= (RooCatType*)_types.At(i);
      os << indent;
      type->printToStream(os,opt,indent);
    }
  }
}


void RooAbsCategory::copyCache(const RooAbsArg* source) 
{
  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  RooAbsCategory* other = dynamic_cast<RooAbsCategory*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  _value = other->_value ;
  setValueDirty() ;
}


void RooAbsCategory::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree
  TString idxName(GetName()) ;
  TString lblName(GetName()) ;  
  idxName.Append("_idx") ;
  lblName.Append("_lbl") ;

  // First determine if branch is taken
  if (t.GetBranch(idxName)) {
    t.SetBranchAddress(idxName,&((Int_t&)_value._value)) ;
  } else {    
    TString format(idxName);
    format.Append("/I");
    void* ptr = &(_value._value) ;
    t.Branch(idxName, ptr, (const Text_t*)format, bufSize);
  }

  // First determine if branch is taken
  if (t.GetBranch(lblName)) {
    t.SetBranchAddress(lblName,_value._label) ;
  } else {    
    TString format(lblName);
    format.Append("/C");
    void* ptr = _value._label ;
    t.Branch(lblName, ptr, (const Text_t*)format, bufSize);
  }
}


const RooCatType* RooAbsCategory::getOrdinal(UInt_t n) const {
  return (const RooCatType*)_types.At(n);
}

RooAbsArg *RooAbsCategory::createFundamental() const {
  // Create a RooCategory fundamental object with our properties.

  // Add and precalculate new category column 
  RooCategory *fund= new RooCategory(GetName(),GetTitle()) ; 

  // Copy states
  TIterator* tIter = typeIterator() ;
  RooCatType* type ;
  while (type=(RooCatType*)tIter->Next()) {
    ((RooAbsCategory*)fund)->defineType(type->GetName(),type->getVal()) ;
  }
  delete tIter;

  return fund;
}


Int_t RooAbsCategory::getPlotBin() const 
{
  //Synchronize _value
  getIndex() ; 

  // Lookup ordinal index number 
  return _types.IndexOf(&_value) ;
}


RooAbsBinIter* RooAbsCategory::createPlotBinIterator() const 
{
  return new RooCatBinIter(*this) ;
}
