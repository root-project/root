/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCategory.cc,v 1.34 2002/04/08 20:20:44 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [CAT] --
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
#include "TLeaf.h"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooCategory.hh"

ClassImp(RooAbsCategory) 
;

RooAbsCategory::RooAbsCategory(const char *name, const char *title) : 
  RooAbsArg(name,title)
{
  // Constructor
  _typeIter = _types.MakeIterator() ;
  setValueDirty() ;  
  setShapeDirty() ;  
}

RooAbsCategory::RooAbsCategory(const RooAbsCategory& other,const char* name) :
  RooAbsArg(other,name), _value(other._value) 
{
  // Copy constructor, copies the registered category states from the original.
  _typeIter = _types.MakeIterator() ;

  other._typeIter->Reset() ;
  TObject* obj ;
  while (obj=other._typeIter->Next()) {
    _types.Add(obj->Clone()) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsCategory::~RooAbsCategory()
{
  // Destructor

  // We own the contents of _types 
  delete _typeIter ;
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
  // Recalculate current value and check validity of new result.

  RooCatType value = evaluate() ;
  
  // Standard tracing code goes here
  if (!isValid(value)) {
  }

  // Call optional subclass tracing code
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
  // Define a new state with given name. The lowest available
  // integer number is assigned as index value

  // Find lowest unused index
  Int_t index(-1) ;
  while(lookupType(++index,kFALSE)) ;
  
  // Assign this index to given label 
  return defineType(label,index) ;
}


const RooCatType* RooAbsCategory::defineTypeUnchecked(const char* label, Int_t index) 
{
  Bool_t first = _types.GetEntries()?kFALSE:kTRUE ;
  RooCatType *newType = new RooCatType(label,index) ;
  _types.Add(newType) ;

  if (first) _value = RooCatType(label,index) ;
  setShapeDirty() ;

  return newType ;  
}



const RooCatType* RooAbsCategory::defineType(const char* label, Int_t index) 
{
  // Define new state with given name and index number.

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

  return defineTypeUnchecked(label,index) ;
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
  _typeIter->Reset() ;
  while(type=(RooCatType*)_typeIter->Next()){
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
  _typeIter->Reset() ;
  while(type=(RooCatType*)_typeIter->Next()){  
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
  _typeIter->Reset() ;
  while(type=(RooCatType*)_typeIter->Next()){  
    if((*type) == label) return type; // delegate comparison to RooCatType
  }

  // Try if label represents integer number
  char* endptr ;
  Int_t idx=strtol(label,&endptr,10)  ;
  if (endptr==label+strlen(label)) {
    _typeIter->Reset() ;
    while(type=(RooCatType*)_typeIter->Next()){  
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
  // Check if given state is defined for this object
  return isValidIndex(value.getVal()) ;
}

Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  // Create a table matching the shape of this category
  return new Roo1DTable(GetName(),label,*this) ;
}

Bool_t RooAbsCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from stream (dummy for now)

  return kFALSE ;
} 

void RooAbsCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to stream 
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
    _typeIter->Reset() ;
    while(type=(RooCatType*)_typeIter->Next()) {
      os << indent;
      type->printToStream(os,opt,indent);
    }
  }
}


void RooAbsCategory::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach the category index and label to as branches
  // to the given TTree. The index field will be attached
  // as integer with name <name>_idx, the label field will be attached
  // as char[] with label <name>_lbl.

  // First check if there is an integer branch matching the category name
  TBranch* branch = t.GetBranch(GetName()) ;
  if (branch) {

    TString typeName(((TLeaf*)branch->GetListOfLeaves()->At(0))->GetTypeName()) ;
    if (!typeName.CompareTo("Int_t")) {
      // Imported TTree: attach only index field as branch

      cout << "RooAbsCategory::attachToTree(" << GetName() << ") TTree branch " << GetName() 
	   << " will be interpreted as category index" << endl ;

      t.SetBranchAddress(GetName(),&((Int_t&)_value._value)) ;
      setAttribute("INTIDXONLY_TREE_BRANCH",kTRUE) ;      
      return ;
    } else if (!typeName.CompareTo("UChar_t")) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree UChar_t branch " << GetName() 
	   << " will be interpreted as category index" << endl ;
      t.SetBranchAddress(GetName(),&((Bool_t&)_value._value)) ;
      setAttribute("UCHARIDXONLY_TREE_BRANCH",kTRUE) ;
      return ;
    } 

    if (branch->GetCompressionLevel()<0) {
      cout << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << GetName() << endl ;
      branch->SetCompressionLevel(1) ;
    }
  }

  // Native TTree: attach both index and label of category as branches  
  TString idxName(GetName()) ;
  TString lblName(GetName()) ;  
  idxName.Append("_idx") ;
  lblName.Append("_lbl") ;
  
  // First determine if branch is taken
  if (branch = t.GetBranch(idxName)) {    

    t.SetBranchAddress(idxName,&((Int_t&)_value._value)) ;
    if (branch->GetCompressionLevel()<0) {
      cout << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << idxName << endl ;
      branch->SetCompressionLevel(1) ;
    }
    
  } else {    
    TString format(idxName);
    format.Append("/I");
    void* ptr = &(_value._value) ;
    branch = t.Branch(idxName, ptr, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(1) ;
  }
  
  // First determine if branch is taken
  if (branch = t.GetBranch(lblName)) {

    t.SetBranchAddress(lblName,_value._label) ;
    if (branch->GetCompressionLevel()<0) {
      cout << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << lblName << endl ;
      branch->SetCompressionLevel(1) ;
    }

  } else {    
    TString format(lblName);
    format.Append("/C");
    void* ptr = _value._label ;
    branch = t.Branch(lblName, ptr, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(1) ;
  }
}


void RooAbsCategory::fillTreeBranch(TTree& t) 
{
  // Attach object to a branch of given TTree

  TString idxName(GetName()) ;
  TString lblName(GetName()) ;  
  idxName.Append("_idx") ;
  lblName.Append("_lbl") ;

  // First determine if branch is taken
  TBranch* idxBranch = t.GetBranch(idxName) ;
  TBranch* lblBranch = t.GetBranch(lblName) ;
  if (!idxBranch||!lblBranch) { 
    cout << "RooAbsCategory::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }

  idxBranch->Fill() ;
  lblBranch->Fill() ;  
}


void RooAbsCategory::copyCache(const RooAbsArg* source) 
{
  // Copy the cached value from given source and raise dirty flag.
  // It is the callers responsability to unsure that the sources
  // cache is clean before this function is called, e.g. by
  // calling syncCache() on the source.

  RooAbsCategory* other = dynamic_cast<RooAbsCategory*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  if (source->getAttribute("INTIDXONLY_TREE_BRANCH")) {
    // Lookup cat state from other-index because label is missing
    const RooCatType* type = lookupType(other->_value._value) ;
    if (type) {
      _value = *type ;
    } else {
      cout << "RooAbsCategory::copyCache(" << GetName() 
	   << ") ERROR: index of source arg " << source->GetName() 
	   << " is invalid (" << other->_value._value 
	   << "), value not updated" << endl ;
    }
  } if (source->getAttribute("UCHARIDXONLY_TREE_BRANCH")) {
    // Lookup cat state from other-index because label is missing
    Int_t tmp = *reinterpret_cast<UChar_t*>(&(other->_value._value)) ;
    const RooCatType* type = lookupType(tmp) ;
    if (type) {
      _value = *type ;
    } else {
      cout << "RooAbsCategory::copyCache(" << GetName() 
	   << ") ERROR: index of source arg " << source->GetName() 
	   << " is invalid (" << tmp
	   << "), value not updated" << endl ;
    }
  } else {
    _value = other->_value ;
  }

  setValueDirty() ;
}


const RooCatType* RooAbsCategory::getOrdinal(UInt_t n) const 
{
  // Return state definition of ordinal nth defined state,
  // needed by the generator mechanism.
  
  return (const RooCatType*)_types.At(n);
}

RooAbsArg *RooAbsCategory::createFundamental(const char* newname) const 
{
  // Create a RooCategory fundamental object with our properties.

  // Add and precalculate new category column 
  RooCategory *fund= new RooCategory(newname?newname:GetName(),GetTitle()) ; 

  // Copy states
  TIterator* tIter = typeIterator() ;
  RooCatType* type ;
  while (type=(RooCatType*)tIter->Next()) {
    ((RooAbsCategory*)fund)->defineType(type->GetName(),type->getVal()) ;
  }
  delete tIter;

  return fund;
}


Bool_t RooAbsCategory::isSignType(Bool_t mustHaveZero) const 
{
  // Determine if category has 2 or 3 states with index values -1,0,1

  if (numTypes()>3||numTypes()<2) return kFALSE ;
  if (mustHaveZero&&numTypes()!=3) return kFALSE ;

  Bool_t ret(kTRUE) ;
  TIterator* tIter = typeIterator() ;
  RooCatType* type ;
  while(type=(RooCatType*)tIter->Next()) {
    if (abs(type->getVal())>1) ret=kFALSE ;
  }
  
  delete tIter ;
  return ret ;
}
