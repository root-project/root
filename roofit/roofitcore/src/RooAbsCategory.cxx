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
// RooAbsCategory is the common abstract base class for objects that
// represent a discrete value with a finite number of states. Each
// state consist of a label/index pair, which is stored in a
// RooCatType object.
// 
// Implementation of RooAbsCategory may be derived, there no interface
// is provided to modify the contents, nor a public interface to define states.
// END_HTML
//
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include "TString.h"
#include "TH1.h"
#include "TTree.h"
#include "TLeaf.h"
#include "RooAbsCategory.h"
#include "RooArgSet.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooMsgService.h"
#include "RooVectorDataStore.h"

ClassImp(RooAbsCategory) 
;


//_____________________________________________________________________________
RooAbsCategory::RooAbsCategory(const char *name, const char *title) : 
  RooAbsArg(name,title), _value("NULL",0), _treeVar(kFALSE)
{
  // Constructor

  _typeIter = _types.MakeIterator() ;
  setValueDirty() ;  
  setShapeDirty() ;  
}



//_____________________________________________________________________________
RooAbsCategory::RooAbsCategory(const RooAbsCategory& other,const char* name) :
  RooAbsArg(other,name), _value(other._value), _treeVar(other._treeVar) 
{
  // Copy constructor, copies the registered category states from the original.

  _typeIter = _types.MakeIterator() ;

  other._typeIter->Reset() ;
  TObject* obj ;
  while ((obj=other._typeIter->Next())) {
    _types.Add(obj->Clone()) ;
  }

  setValueDirty() ;
  setShapeDirty() ;
}



//_____________________________________________________________________________
RooAbsCategory::~RooAbsCategory()
{
  // Destructor

  // We own the contents of _types 
  delete _typeIter ;
  _types.Delete() ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
TIterator* RooAbsCategory::typeIterator() const
{
  // Return iterator over all defined states

  return _types.MakeIterator() ;
}


//_____________________________________________________________________________
Bool_t RooAbsCategory::operator==(Int_t index) const
{
  // Equality operator with a integer (compares with state index number)

  return (index==getIndex()) ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::operator==(const char* label) const
{
  // Equality operator with a string (compares with state label string)

  return !TString(label).CompareTo(getLabel()) ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::operator==(const RooAbsArg& other) 
{
  // Equality operator with another RooAbsArg. Only functional
  // is also a RooAbsCategory, will return true if index is the same

  const RooAbsCategory* otherCat = dynamic_cast<const RooAbsCategory*>(&other) ;
  return otherCat ? operator==(otherCat->getIndex()) : kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::isValidIndex(Int_t index) const
{
  // Check if state with given index is defined

  return lookupType(index,kFALSE)?kTRUE:kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::isValidLabel(const char* label) const
{
  // Check if state with given name is defined

  return lookupType(label)?kTRUE:kFALSE ;
}



//_____________________________________________________________________________
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


//_____________________________________________________________________________
const RooCatType* RooAbsCategory::defineTypeUnchecked(const char* label, Int_t index) 
{
  // Internal version of defineType that does not check if type
  // already exists

  Bool_t first = _types.GetEntries()?kFALSE:kTRUE ;
  RooCatType *newType = new RooCatType(label,index) ;
  _types.Add(newType) ;

  if (first) _value = RooCatType(label,index) ;
  setShapeDirty() ;

  return newType ;  
}



//_____________________________________________________________________________
const RooCatType* RooAbsCategory::defineType(const char* label, Int_t index) 
{
  // Define new state with given name and index number.

  if (isValidIndex(index)) {
    coutE(InputArguments) << "RooAbsCategory::defineType(" << GetName() << "): index " 
			  << index << " already assigned" << endl ;
    return 0 ;
  }

  if (isValidLabel(label)) {
    coutE(InputArguments) << "RooAbsCategory::defineType(" << GetName() << "): label " 
			  << label << " already assigned or not allowed" << endl ;
    return 0 ;
  }

  return defineTypeUnchecked(label,index) ;
}



//_____________________________________________________________________________
void RooAbsCategory::clearTypes() 
{
  // Delete all currently defined states

  _types.Delete() ;
  _value = RooCatType("",0) ;
  setShapeDirty() ;
}



//_____________________________________________________________________________
const RooCatType* RooAbsCategory::lookupType(const RooCatType &other, Bool_t printError) const 
{
  // Find our type that matches the specified type, or return 0 for no match.

  RooCatType* type ;
  _typeIter->Reset() ;
  while((type=(RooCatType*)_typeIter->Next())){
    if((*type) == other) return type; // delegate comparison to RooCatType
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for ";
    if (dologE(InputArguments)) {
      other.printStream(ccoutE(InputArguments),kName|kValue,kSingleLine);
    }
  }
  return 0 ;
}



//_____________________________________________________________________________
const RooCatType* RooAbsCategory::lookupType(Int_t index, Bool_t printError) const
{
  // Find our type corresponding to the specified index, or return 0 for no match.

  RooCatType* type ;
  _typeIter->Reset() ;
  while((type=(RooCatType*)_typeIter->Next())){  
    if((*type) == index) return type; // delegate comparison to RooCatType
  }
  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for index "
			  << index << endl;
  }
  return 0 ;
}



//_____________________________________________________________________________
const RooCatType* RooAbsCategory::lookupType(const char* label, Bool_t printError) const 
{
  // Find our type corresponding to the specified label, or return 0 for no match.
 
  RooCatType* type ;
  _typeIter->Reset() ;
  while((type=(RooCatType*)_typeIter->Next())){  
    if((*type) == label) return type; // delegate comparison to RooCatType
  }

  // Try if label represents integer number
  char* endptr ;
  Int_t idx=strtol(label,&endptr,10)  ;
  if (endptr==label+strlen(label)) {
    _typeIter->Reset() ;
    while((type=(RooCatType*)_typeIter->Next())){  
       if((*type) == idx) return type; // delegate comparison to RooCatType
     }
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for label "
			  << label << endl;
  }
  return 0 ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::isValid() const
{
  // Check if current value is a valid state

  return isValid(_value) ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::isValid(const RooCatType& value)  const
{
  // Check if given state is defined for this object

  return isValidIndex(value.getVal()) ;
}



//_____________________________________________________________________________
Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  // Create a table matching the shape of this category

  return new Roo1DTable(GetName(),label,*this) ;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::readFromStream(istream&, Bool_t, Bool_t) 
{
  // Read object contents from stream (dummy for now)

  return kFALSE ;
} 



//_____________________________________________________________________________
void RooAbsCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to ostream 

  if (compact) {
    os << getLabel() ;
  } else {
    os << getLabel() ;
  }
}



//_____________________________________________________________________________
void RooAbsCategory::printValue(ostream& os) const
{
  // Print value (label name)
  os << getLabel() ;
}



//_____________________________________________________________________________
void RooAbsCategory::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printStream() we add:
  //
  //     Shape : label, index, defined types

  RooAbsArg::printMultiline(os,contents,verbose,indent);

  os << indent << "--- RooAbsCategory ---" << endl;
  if (_types.GetEntries()==0) {
    os << indent << "  ** No values defined **" << endl;
    return;
  }
  os << indent << "  Value is \"" << getLabel() << "\" (" << getIndex() << ")" << endl;
  os << indent << "  Has the following possible values:" << endl;
  indent.Append("    ");
  RooCatType *type;
  _typeIter->Reset() ;
  while((type=(RooCatType*)_typeIter->Next())) {
    os << indent;
    type->printStream(os,kName|kValue,kSingleLine,indent);
  }
}



//_____________________________________________________________________________
void RooAbsCategory::attachToVStore(RooVectorDataStore& vstore)
{
  // Attach the category index and label to as branches to the given vector store
  RooVectorDataStore::CatVector* cv = vstore.addCategory(this) ;
  cv->setBuffer(&_value) ;  
}




//_____________________________________________________________________________
void RooAbsCategory::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach the category index and label to as branches to the given
  // TTree. The index field will be attached as integer with name
  // <name>_idx, the label field will be attached as char[] with label
  // <name>_lbl.

  // First check if there is an integer branch matching the category name
  TString cleanName(cleanBranchName()) ;
  TBranch* branch = t.GetBranch(cleanName) ;
  if (branch) {

    TString typeName(((TLeaf*)branch->GetListOfLeaves()->At(0))->GetTypeName()) ;
    if (!typeName.CompareTo("Int_t")) {
      // Imported TTree: attach only index field as branch

      coutI(DataHandling) << "RooAbsCategory::attachToTree(" << GetName() << ") TTree branch " << GetName() 
			  << " will be interpreted as category index" << endl ;
      
      t.SetBranchAddress(cleanName,&((Int_t&)_value._value)) ;
      setAttribute("INTIDXONLY_TREE_BRANCH",kTRUE) ;      
      _treeVar = kTRUE ;
      return ;
    } else if (!typeName.CompareTo("UChar_t")) {
      coutI(DataHandling) << "RooAbsReal::attachToTree(" << GetName() << ") TTree UChar_t branch " << GetName() 
			  << " will be interpreted as category index" << endl ;
      t.SetBranchAddress(cleanName,&_byteValue) ;
      setAttribute("UCHARIDXONLY_TREE_BRANCH",kTRUE) ;
      _treeVar = kTRUE ;
      return ;
    } 

    if (branch->GetCompressionLevel()<0) {
      cxcoutD(DataHandling) << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << GetName() << endl ;
      branch->SetCompressionLevel(1) ;
    }
  }

  // Native TTree: attach both index and label of category as branches  
  TString idxName(cleanName) ;
  TString lblName(cleanName) ;  
  idxName.Append("_idx") ;
  lblName.Append("_lbl") ;
  
  // First determine if branch is taken
  if ((branch = t.GetBranch(idxName))) {    

    t.SetBranchAddress(idxName,&((Int_t&)_value._value)) ;
    if (branch->GetCompressionLevel()<0) {
      cxcoutD(Contents) << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << idxName << endl ;
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
  if ((branch = t.GetBranch(lblName))) {

    t.SetBranchAddress(lblName,_value._label) ;
    if (branch->GetCompressionLevel()<0) {
      cxcoutD(DataHandling) << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << lblName << endl ;
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



//_____________________________________________________________________________
void RooAbsCategory::fillTreeBranch(TTree& t) 
{
  // Fill tree branches associated with current object with current value

  TString idxName(GetName()) ;
  TString lblName(GetName()) ;  
  idxName.Append("_idx") ;
  lblName.Append("_lbl") ;

  // First determine if branch is taken
  TBranch* idxBranch = t.GetBranch(idxName) ;
  TBranch* lblBranch = t.GetBranch(lblName) ;
  if (!idxBranch||!lblBranch) { 
    coutF(DataHandling) << "RooAbsCategory::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }

  idxBranch->Fill() ;
  lblBranch->Fill() ;  
}



//_____________________________________________________________________________
void RooAbsCategory::setTreeBranchStatus(TTree& t, Bool_t active) 
{
  // (De)activate associate tree branch

  TBranch* branch = t.GetBranch(Form("%s_idx",GetName())) ;
  if (branch) { 
    t.SetBranchStatus(Form("%s_idx",GetName()),active?1:0) ;
    t.SetBranchStatus(Form("%s_lbl",GetName()),active?1:0) ;
  }
}



//_____________________________________________________________________________
void RooAbsCategory::syncCache(const RooArgSet*) 
{ 
  // Explicitly synchronize RooAbsCategory internal cache

  getIndex() ; 
}



//_____________________________________________________________________________
void RooAbsCategory::copyCache(const RooAbsArg* source, Bool_t /*valueOnly*/) 
{
  // Copy the cached value from given source and raise dirty flag.
  // It is the callers responsability to ensure that the sources
  // cache is clean(valid) before this function is called, e.g. by
  // calling syncCache() on the source.

  RooAbsCategory* other = static_cast<RooAbsCategory*>(const_cast<RooAbsArg*>(source)) ;

  if (!_treeVar) {
    _value = other->_value ;
  } else {
    if (source->getAttribute("INTIDXONLY_TREE_BRANCH")) {
      // Lookup cat state from other-index because label is missing
      const RooCatType* type = lookupType(other->_value._value) ;
      if (type) {
	_value = *type ;
      } else {
	coutE(DataHandling) << "RooAbsCategory::copyCache(" << GetName() 
			    << ") ERROR: index of source arg " << source->GetName() 
			    << " is invalid (" << other->_value._value 
			    << "), value not updated" << endl ;
      }
    } if (source->getAttribute("UCHARIDXONLY_TREE_BRANCH")) {
      // Lookup cat state from other-index because label is missing
      Int_t tmp = other->_byteValue ;
      const RooCatType* type = lookupType(tmp) ;
      if (type) {
	_value = *type ;
      } else {
	coutE(DataHandling) << "RooAbsCategory::copyCache(" << GetName() 
			    << ") ERROR: index of source arg " << source->GetName() 
			    << " is invalid (" << tmp
			    << "), value not updated" << endl ;
      }
    } 
  }

  setValueDirty() ;
}



//_____________________________________________________________________________
const RooCatType* RooAbsCategory::getOrdinal(UInt_t n, const char* /*rangeName*/) const 
{
  // Return state definition of ordinal nth defined state,
  // needed by the generator mechanism.
  
  return (const RooCatType*)_types.At(n);
}



//_____________________________________________________________________________
RooAbsArg *RooAbsCategory::createFundamental(const char* newname) const 
{
  // Create a RooCategory fundamental object with our properties.

  // Add and precalculate new category column 
  RooCategory *fund= new RooCategory(newname?newname:GetName(),GetTitle()) ; 

  // Copy states
  TIterator* tIter = typeIterator() ;
  RooCatType* type ;
  while ((type=(RooCatType*)tIter->Next())) {
    ((RooAbsCategory*)fund)->defineType(type->GetName(),type->getVal()) ;
  }
  delete tIter;

  return fund;
}



//_____________________________________________________________________________
Bool_t RooAbsCategory::isSignType(Bool_t mustHaveZero) const 
{
  // Determine if category has 2 or 3 states with index values -1,0,1

  if (numTypes()>3||numTypes()<2) return kFALSE ;
  if (mustHaveZero&&numTypes()!=3) return kFALSE ;

  Bool_t ret(kTRUE) ;
  TIterator* tIter = typeIterator() ;
  RooCatType* type ;
  while((type=(RooCatType*)tIter->Next())) {
    if (abs(type->getVal())>1) ret=kFALSE ;
  }
  
  delete tIter ;
  return ret ;
}
