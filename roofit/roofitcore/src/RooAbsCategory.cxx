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
\file RooAbsCategory.cxx
\class RooAbsCategory
\ingroup Roofitcore

RooAbsCategory is the common abstract base class for objects that
represent a discrete value with a finite number of states. Each
state consist of a label/index pair, which is stored in a
RooCatType object.

Implementation of RooAbsCategory may be derived, there no interface
is provided to modify the contents, nor a public interface to define states.
**/

#include "RooFit.h"

#include "Compression.h"
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
#include "RooLinkedListIter.h"

using namespace std;

ClassImp(RooAbsCategory);



////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCategory::RooAbsCategory(const char *name, const char *title) :
  RooAbsArg(name,title), _byteValue(0), _value("NULL",0), _treeVar(kFALSE)
{
  setValueDirty() ;
  setShapeDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor, copies the registered category states from the original.

RooAbsCategory::RooAbsCategory(const RooAbsCategory& other,const char* name) :
  RooAbsArg(other,name),  _byteValue(0), _value(other._value), _treeVar(other._treeVar)
{
  for (const auto cat : other._types) {
    _types.push_back(new RooCatType(*cat));
  }

  setValueDirty() ;
  setShapeDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsCategory::~RooAbsCategory()
{

}



////////////////////////////////////////////////////////////////////////////////
/// Return index number of current state

RooAbsCategory::value_type RooAbsCategory::getIndex() const
{
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    clearValueDirty() ;
    clearShapeDirty() ;
  }

  return _value.getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return label string of current state

const char* RooAbsCategory::getLabel() const
{
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;

    clearValueDirty() ;
    clearShapeDirty() ;
  }

  const char* ret = _value.GetName() ;
  // If label is not set, do it now on the fly
  if (ret==0) {
    _value.SetName(lookupType(_value.getVal())->GetName()) ;
  }
  return _value.GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Recalculate current value and check validity of new result.

RooCatType RooAbsCategory::traceEval() const
{
  RooCatType value = evaluate() ;

  // Standard tracing code goes here
  if (!isValid(value)) {
  }

  // Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


////////////////////////////////////////////////////////////////////////////////
/// Equality operator with a integer (compares with state index number)

Bool_t RooAbsCategory::operator==(RooAbsCategory::value_type index) const
{
  return (index==getIndex()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator with a string (compares with state label string)

Bool_t RooAbsCategory::operator==(const char* label) const
{
  return strcmp(label, getLabel()) == 0;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator with another RooAbsArg. Only functional
/// is also a RooAbsCategory, will return true if index is the same

Bool_t RooAbsCategory::operator==(const RooAbsArg& other) const
{
  const RooAbsCategory* otherCat = dynamic_cast<const RooAbsCategory*>(&other) ;
  return otherCat ? operator==(otherCat->getIndex()) : kFALSE ;
}


////////////////////////////////////////////////////////////////////////////////

Bool_t RooAbsCategory::isIdentical(const RooAbsArg& other, Bool_t assumeSameType) const
{
  if (!assumeSameType) {
    const RooAbsCategory* otherCat = dynamic_cast<const RooAbsCategory*>(&other) ;
    return otherCat ? operator==(otherCat->getIndex()) : kFALSE ;
  } else {
    return getIndex() == static_cast<const RooAbsCategory&>(other).getIndex();
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Check if state with given index is defined

Bool_t RooAbsCategory::isValidIndex(RooAbsCategory::value_type index) const
{
  return lookupType(index,kFALSE)?kTRUE:kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if state with given name is defined

Bool_t RooAbsCategory::isValidLabel(const char* label) const
{
  return lookupType(label)?kTRUE:kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define a new state with given name. The lowest available
/// integer number is assigned as index value

const RooCatType* RooAbsCategory::defineType(const char* label)
{
  // Find lowest unused index
  RooAbsCategory::value_type index(-1) ;
  while(lookupType(++index,kFALSE)) ;

  // Assign this index to given label
  return defineType(label,index) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Internal version of defineType that does not check if type
/// already exists

const RooCatType* RooAbsCategory::defineTypeUnchecked(const char* label, RooAbsCategory::value_type index)
{
  _types.push_back(new RooCatType(label, index));

  if (_types.size() == 1) _value = *_types.back();
  setShapeDirty();

  return _types.back();
}



////////////////////////////////////////////////////////////////////////////////
/// Define new state with given name and index number.

const RooCatType* RooAbsCategory::defineType(const char* label, RooAbsCategory::value_type index)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Delete all currently defined states

void RooAbsCategory::clearTypes()
{
  for (auto cat : _types)
    delete cat;
  _types.clear();
  _value = RooCatType("",0) ;
  setShapeDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find our type that matches the specified type, or return 0 for no match.

const RooCatType* RooAbsCategory::lookupType(const RooCatType &other, Bool_t printError) const
{
  for (const auto type : _types) {
    if (*type == other) return type; // delegate comparison to RooCatType
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for ";
    if (dologE(InputArguments)) {
      other.printStream(ccoutE(InputArguments),kName|kValue,kSingleLine);
    }
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find our type corresponding to the specified index, or return 0 for no match.

const RooCatType* RooAbsCategory::lookupType(RooAbsCategory::value_type index, Bool_t printError) const
{
  for (const auto type : _types) {
    if(*type == index) return type; // delegate comparison to RooCatType
  }
  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for index "
			  << index << endl;
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find our type corresponding to the specified label, or return 0 for no match.

const RooCatType* RooAbsCategory::lookupType(const char* label, Bool_t printError) const
{
  for (const auto type : _types) {
    if(*type == label) return type; // delegate comparison to RooCatType
  }

  // Try if label represents integer number
  char* endptr ;
  RooAbsCategory::value_type idx=strtol(label,&endptr,10)  ;
  if (endptr==label+strlen(label)) {
    for (const auto type : _types) {
       if(*type == idx) return type; // delegate comparison to RooCatType
     }
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for label "
			  << label << endl;
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if current value is a valid state

Bool_t RooAbsCategory::isValid() const
{
  return isValid(_value) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if given state is defined for this object

Bool_t RooAbsCategory::isValid(const RooCatType& value)  const
{
  return isValidIndex(value.getVal()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a table matching the shape of this category

Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  return new Roo1DTable(GetName(),label,*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Read object contents from stream (dummy for now)

Bool_t RooAbsCategory::readFromStream(istream&, Bool_t, Bool_t)
{
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write object contents to ostream

void RooAbsCategory::writeToStream(ostream& os, Bool_t /* compact */) const
{
  os << getLabel() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print value (label name)

void RooAbsCategory::printValue(ostream& os) const
{
  os << getLabel() << "(idx = " << getIndex() << ")" << endl ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream. In addition to the info
/// from RooAbsArg::printStream() we add:
///
///     Shape : label, index, defined types

void RooAbsCategory::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  RooAbsArg::printMultiline(os,contents,verbose,indent);

  os << indent << "--- RooAbsCategory ---" << endl;
  if (_types.empty()) {
    os << indent << "  ** No values defined **" << endl;
    return;
  }
  os << indent << "  Value is \"" << getLabel() << "\" (" << getIndex() << ")" << endl;
  os << indent << "  Has the following possible values:" << endl;
  indent.Append("    ");
  for (const auto type : _types) {
    os << indent;
    type->printStream(os,kName|kValue,kSingleLine,indent);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Attach the category index and label to as branches to the given vector store

void RooAbsCategory::attachToVStore(RooVectorDataStore& vstore)
{
  RooVectorDataStore::CatVector* cv = vstore.addCategory(this) ;
  cv->setBuffer(&_value) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Attach the category index and label to as branches to the given
/// TTree. The index field will be attached as integer with name
/// <name>_idx, the label field will be attached as char[] with label
/// <name>_lbl.

void RooAbsCategory::attachToTree(TTree& t, Int_t bufSize)
{
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
      branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
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
      branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
    }

  } else {
    TString format(idxName);
    format.Append("/I");
    void* ptr = &(_value._value) ;
    branch = t.Branch(idxName, ptr, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
  }

  // First determine if branch is taken
  if ((branch = t.GetBranch(lblName))) {

    t.SetBranchAddress(lblName,_value._label) ;
    if (branch->GetCompressionLevel()<0) {
      cxcoutD(DataHandling) << "RooAbsCategory::attachToTree(" << GetName() << ") Fixing compression level of branch " << lblName << endl ;
      branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
    }

  } else {
    TString format(lblName);
    format.Append("/C");
    void* ptr = _value._label ;
    branch = t.Branch(lblName, ptr, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(ROOT::RCompressionSetting::EDefaults::kUseGlobal % 100) ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Fill tree branches associated with current object with current value

void RooAbsCategory::fillTreeBranch(TTree& t)
{
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



////////////////////////////////////////////////////////////////////////////////
/// (De)activate associate tree branch

void RooAbsCategory::setTreeBranchStatus(TTree& t, Bool_t active)
{
  TBranch* branch = t.GetBranch(Form("%s_idx",GetName())) ;
  if (branch) {
    t.SetBranchStatus(Form("%s_idx",GetName()),active?1:0) ;
    t.SetBranchStatus(Form("%s_lbl",GetName()),active?1:0) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Explicitly synchronize RooAbsCategory internal cache

void RooAbsCategory::syncCache(const RooArgSet*)
{
  getIndex() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy the cached value from given source and raise dirty flag.
/// It is the callers responsability to ensure that the sources
/// cache is clean(valid) before this function is called, e.g. by
/// calling syncCache() on the source.

void RooAbsCategory::copyCache(const RooAbsArg *source, Bool_t /*valueOnly*/, Bool_t setValDirty)
{
   RooAbsCategory *other = static_cast<RooAbsCategory *>(const_cast<RooAbsArg *>(source));

   if (!_treeVar) {
      _value = other->_value;
   } else {
      if (source->getAttribute("INTIDXONLY_TREE_BRANCH")) {
         // Lookup cat state from other-index because label is missing
         const RooCatType *type = lookupType(other->_value._value);
         if (type) {
            _value = *type;
         } else {
            coutE(DataHandling) << "RooAbsCategory::copyCache(" << GetName() << ") ERROR: index of source arg "
                                << source->GetName() << " is invalid (" << other->_value._value
                                << "), value not updated" << endl;
         }
      } else if (source->getAttribute("UCHARIDXONLY_TREE_BRANCH")) {
         // Lookup cat state from other-index because label is missing
         Int_t tmp = other->_byteValue;
         const RooCatType *type = lookupType(tmp);
         if (type) {
            _value = *type;
         } else {
            coutE(DataHandling) << "RooAbsCategory::copyCache(" << GetName() << ") ERROR: index of source arg "
                                << source->GetName() << " is invalid (" << tmp << "), value not updated" << endl;
         }
      }
   }

   if (setValDirty) {
      setValueDirty();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return state definition of ordinal nth defined state,
/// needed by the generator mechanism.

const RooCatType* RooAbsCategory::getOrdinal(UInt_t n, const char* /*rangeName*/) const
{
  return _types.size() > n ? _types[n] : nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooCategory fundamental object with our properties.

RooAbsArg *RooAbsCategory::createFundamental(const char* newname) const
{
  // Add and precalculate new category column
  RooCategory *fund= new RooCategory(newname?newname:GetName(),GetTitle()) ;

  // Copy states
  for (const auto type : _types) {
    ((RooAbsCategory*)fund)->defineType(type->GetName(), type->getVal());
  }

  return fund;
}



////////////////////////////////////////////////////////////////////////////////
/// Determine if category has 2 or 3 states with index values -1,0,1

Bool_t RooAbsCategory::isSignType(Bool_t mustHaveZero) const
{
  if (numTypes()>3||numTypes()<2) return kFALSE ;
  if (mustHaveZero&&numTypes()!=3) return kFALSE ;

  Bool_t ret(kTRUE) ;
  for (const auto type : _types) {
    if (abs(type->getVal())>1) ret=kFALSE ;
  }

  return ret ;
}
