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

RooAbsCategory is the base class for objects that represent a discrete value with a finite number of states.

Each state is denoted by an integer and a name. Both can be used to retrieve and
set states, but referring to states by index is more efficient. Conversion between
index and name can be done using lookupName() or lookupIndex().
It is possible to iterate through all defined states using begin() and end().

For category classes deriving from RooAbsCategory, states can only be evaluated, *i.e.*, queried.
Refer to RooAbsCategoryLValue and its derived classes for categories where states can also be set. The
simplest category class whose states can be set, queried and saved in a dataset, refer to RooCategory.

### Interface change in ROOT-6.22
Category data were based in the class RooCatType, holding an index state and a category name truncated to 256
characters. This wastes 64 bytes of storage space per entry, and prevents fast retrieval of category data.
Since ROOT-6.22, categories are only represented by an integer. RooAbsCategory::lookupName() can be used to
retrieve the corresponding state name. There is no limit for the length of the state name.

To not break old code, the old RooCatType interfaces are still available. Whenever possible,
the following replacements should be used:
- lookupType() \f$ \rightarrow \f$ lookupName() / lookupIndex()
- typeIterator() \f$ \rightarrow \f$ range-based for loop / begin() / end()
- isValidIndex(Int_t index) \f$ \rightarrow \f$ hasIndex()
- isValid(const RooCatType&) \f$ \rightarrow \f$ hasIndex() / hasLabel()
**/

#include "RooAbsCategory.h"

#include "RooArgSet.h"
#include "Roo1DTable.h"
#include "RooCategory.h"
#include "RooMsgService.h"
#include "RooVectorDataStore.h"
#include "RooFitLegacy/RooAbsCategoryLegacyIterator.h"
#include "TreeReadBuffer.h"

#include "Compression.h"
#include "TString.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TBranch.h"

#include <functional>
#include <memory>

ClassImp(RooAbsCategory);

/// A category state to signify an invalid category. The category name is empty,
/// the index is the minimal int.
const decltype(RooAbsCategory::_stateNames)::value_type& RooAbsCategory::invalidCategory() {
  static const decltype(RooAbsCategory::_stateNames)::value_type invalid{"", std::numeric_limits<value_type>::min()};
  return invalid;
}


RooAbsCategory::RooAbsCategory() {}


////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooAbsCategory::RooAbsCategory(const char *name, const char *title) :
  RooAbsArg(name,title), _currentIndex(0)
{
  setValueDirty() ;
  setShapeDirty() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor, copies the registered category states from the original.

RooAbsCategory::RooAbsCategory(const RooAbsCategory& other,const char* name) :
  RooAbsArg(other,name),  _currentIndex(other._currentIndex),
  _stateNames(other._stateNames),
  _insertionOrder(other._insertionOrder)
{
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

RooAbsCategory::value_type RooAbsCategory::getCurrentIndex() const
{
  if (isValueDirty() || isShapeDirty()) {
    _currentIndex = evaluate();

    clearValueDirty() ;
  }

  return _currentIndex;
}



////////////////////////////////////////////////////////////////////////////////
/// Return label string of current state.

const char* RooAbsCategory::getCurrentLabel() const
{
  const auto index = getCurrentIndex();
  for (const auto& item : stateNames()) {
    if (item.second == index)
      return item.first.c_str();
  }

  return "";
}


////////////////////////////////////////////////////////////////////////////////
/// Equality operator with a integer (compares with state index number)

bool RooAbsCategory::operator==(RooAbsCategory::value_type index) const
{
  return (index==getCurrentIndex()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator with a string (compares with state label string)

bool RooAbsCategory::operator==(const char* label) const
{
  return strcmp(label, getCurrentLabel()) == 0;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator with another RooAbsArg. Only functional
/// is also a RooAbsCategory, will return true if index is the same

bool RooAbsCategory::operator==(const RooAbsArg& other) const
{
  const RooAbsCategory* otherCat = dynamic_cast<const RooAbsCategory*>(&other) ;
  return otherCat ? operator==(otherCat->getCurrentIndex()) : false ;
}


////////////////////////////////////////////////////////////////////////////////

bool RooAbsCategory::isIdentical(const RooAbsArg& other, bool assumeSameType) const
{
  if (!assumeSameType) {
    const RooAbsCategory* otherCat = dynamic_cast<const RooAbsCategory*>(&other) ;
    return otherCat ? operator==(otherCat->getCurrentIndex()) : false ;
  } else {
    return getCurrentIndex() == static_cast<const RooAbsCategory&>(other).getCurrentIndex();
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Check if a state with index `index` exists.
bool RooAbsCategory::hasIndex(RooAbsCategory::value_type index) const
{
  for (const auto& item : stateNames()) {
    if (item.second == index)
      return true;
  }

  return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Look up the name corresponding to the given index.
const std::string& RooAbsCategory::lookupName(value_type index) const {
  for (const auto& item : stateNames()) {
    if (item.second == index)
      return item.first;
  }

  return invalidCategory().first;
}

////////////////////////////////////////////////////////////////////////////////
/// Define a new state with given label. The next available
/// integer is assigned as index value.
const std::map<std::string, RooAbsCategory::value_type>::value_type& RooAbsCategory::defineState(const std::string& label)
{
  return defineState(label, nextAvailableStateIndex());
}


////////////////////////////////////////////////////////////////////////////////
/// Internal version of defineState() that does not check if type
/// already exists
void RooAbsCategory::defineStateUnchecked(const std::string& label, RooAbsCategory::value_type index)
{
  _stateNames.emplace(label, index);
  _insertionOrder.push_back(label);

  if (_stateNames.size() == 1)
    _currentIndex = index;

  setShapeDirty();
}



////////////////////////////////////////////////////////////////////////////////
/// Define new state with given name and index number.

const std::map<std::string, RooAbsCategory::value_type>::value_type& RooAbsCategory::defineState(const std::string& label, RooAbsCategory::value_type index)
{
  auto& theStateNames = stateNames();

  if (hasIndex(index)) {
    coutE(InputArguments) << "RooAbsCategory::" << __func__ << "(" << GetName() << "): index "
           << index << " already assigned" << std::endl;
    return invalidCategory();
  }

  if (hasLabel(label)) {
    coutE(InputArguments) << "RooAbsCategory::" << __func__ << "(" << GetName() << "): label "
           << label << " already assigned or not allowed" << std::endl;
    return invalidCategory();
  }

  const auto result = theStateNames.emplace(label, index);
  _insertionOrder.push_back(label);

  if (theStateNames.size() == 1)
    _currentIndex = index;

  setShapeDirty();

  return *(result.first);
}



////////////////////////////////////////////////////////////////////////////////
/// Delete all currently defined states

void RooAbsCategory::clearTypes()
{
  _stateNames.clear();
  _insertionOrder.clear();
  _currentIndex = invalidCategory().second;
  setShapeDirty() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Find the index number corresponding to the state name.
/// \see hasLabel() for checking if a given label has been defined.
/// \return Index of the category or std::numeric_limits<int>::min() on failure.
RooAbsCategory::value_type RooAbsCategory::lookupIndex(const std::string& stateName) const {
  const auto item = stateNames().find(stateName);
  if (item != stateNames().end()) {
    return item->second;
  }

  return invalidCategory().second;
}

////////////////////////////////////////////////////////////////////////////////
/// Find our type that matches the specified type, or return 0 for no match.
/// \deprecated RooCatType is not used, any more. This function will create one and let it leak.
/// Use lookupIndex() (preferred) or lookupName() instead.
const RooCatType* RooAbsCategory::lookupType(const RooCatType &other, bool printError) const
{
  return lookupType(other.getVal(), printError);
}



////////////////////////////////////////////////////////////////////////////////
/// Find our type corresponding to the specified index, or return nullptr for no match.
/// \deprecated RooCatType is not used, any more. This function will create one and let it leak.
/// Use lookupIndex() (preferred) or lookupName() instead.
const RooCatType* RooAbsCategory::lookupType(RooAbsCategory::value_type index, bool printError) const
{
  for (const auto& item : stateNames())
  if (item.second == index) {
    return retrieveLegacyState(index);
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for index "
        << index << std::endl;
  }

  return nullptr;
}



////////////////////////////////////////////////////////////////////////////////
/// Find our type corresponding to the specified label, or return 0 for no match.
/// \deprecated RooCatType is not used, any more. This function will create one and let it leak.
/// Use lookupIndex() (preferred) or lookupName() instead.
const RooCatType* RooAbsCategory::lookupType(const char* label, bool printError) const
{
  for (const auto& type : stateNames()) {
    if(type.first == label)
      return retrieveLegacyState(type.second);
  }

  // Try if label represents integer number
  char* endptr ;
  RooAbsCategory::value_type idx=strtol(label,&endptr,10)  ;
  if (endptr==label+strlen(label)) {
    return lookupType(idx);
  }

  if (printError) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":lookupType: no match for label "
           << label << std::endl;
  }
  return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Check if given state is defined for this object

bool RooAbsCategory::isValid(const RooCatType& value)  const
{
  return hasIndex(value.getVal()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a table matching the shape of this category

Roo1DTable* RooAbsCategory::createTable(const char *label)  const
{
  return new Roo1DTable(GetName(),label,*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Read object contents from stream (dummy for now)

bool RooAbsCategory::readFromStream(std::istream&, bool, bool)
{
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Write object contents to ostream

void RooAbsCategory::writeToStream(std::ostream& os, bool /* compact */) const
{
  os << getCurrentLabel() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print value (label name)

void RooAbsCategory::printValue(std::ostream& os) const
{
  os << getCurrentLabel() << "(idx = " << getCurrentIndex() << ")" << std::endl;
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream. In addition to the info
/// from RooAbsArg::printStream() we add:
///
///     Shape : label, index, defined types

void RooAbsCategory::printMultiline(std::ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooAbsArg::printMultiline(os,contents,verbose,indent);

  os << indent << "--- RooAbsCategory ---" << std::endl;
  if (stateNames().empty()) {
    os << indent << "  ** No values defined **" << std::endl;
    return;
  }
  os << indent << "  Value = " << getCurrentIndex() << " \"" << getCurrentLabel() << ')' << std::endl;
  os << indent << "  Possible states:" << std::endl;
  indent.Append("    ");
  for (const auto& type : stateNames()) {
    os << indent << type.first << '\t' << type.second << "\n";
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Attach the category index and label to as branches to the given vector store

void RooAbsCategory::attachToVStore(RooVectorDataStore& vstore)
{
  RooVectorDataStore::CatVector* cv = vstore.addCategory(this) ;
  cv->setBuffer(&_currentIndex);
}




////////////////////////////////////////////////////////////////////////////////
/// Attach the category index and label as branches to the given
/// TTree. The index field will be attached as integer with name
/// `<name>_idx`. If a branch `<name>` exists, it attaches to this branch.
void RooAbsCategory::attachToTree(TTree& tree, Int_t bufSize)
{
  // First check if there is an integer branch matching the category name
  std::string cleanName = cleanBranchName().Data();
  TBranch* branch = tree.GetBranch(cleanName.c_str());
  if (!branch) {
    cleanName += "_idx";
    branch = tree.GetBranch(cleanName.c_str());
  }

  if (branch) {
    TLeaf* leaf = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0));

    // Check that leaf is _not_ an array
    Int_t dummy ;
    TLeaf* counterLeaf = leaf->GetLeafCounter(dummy) ;
    if (counterLeaf) {
      coutE(Eval) << "RooAbsCategory::attachToTree(" << GetName() << ") ERROR: TTree branch " << GetName()
        << " is an array and cannot be attached to a RooAbsCategory" << std::endl;
      return ;
    }

    const std::string typeName = leaf->GetTypeName();


    // For different type names, store a function to attach
    std::map<std::string, std::function<std::unique_ptr<TreeReadBuffer>()>> typeMap {
      {"Float_t",   [&](){ return createTreeReadBuffer<Float_t  >(cleanName, tree); }},
      {"Double_t",  [&](){ return createTreeReadBuffer<Double_t >(cleanName, tree); }},
      {"UChar_t",   [&](){ return createTreeReadBuffer<UChar_t  >(cleanName, tree); }},
      {"Boolt_",    [&](){ return createTreeReadBuffer<Bool_t   >(cleanName, tree); }},
      {"Char_t",    [&](){ return createTreeReadBuffer<Char_t   >(cleanName, tree); }},
      {"UInt_t",    [&](){ return createTreeReadBuffer<UInt_t   >(cleanName, tree); }},
      {"Long64_t",  [&](){ return createTreeReadBuffer<Long64_t >(cleanName, tree); }},
      {"ULong64_t", [&](){ return createTreeReadBuffer<ULong64_t>(cleanName, tree); }},
      {"Short_t",   [&](){ return createTreeReadBuffer<Short_t  >(cleanName, tree); }},
      {"UShort_t",  [&](){ return createTreeReadBuffer<UShort_t >(cleanName, tree); }},
    };

    auto typeDetails = typeMap.find(typeName);
    if (typeDetails != typeMap.end()) {
      coutI(DataHandling) << "RooAbsCategory::attachToTree(" << GetName() << ") TTree " << typeName << " branch \"" << cleanName
                  << "\" will be converted to int." << std::endl;
      _treeReadBuffer = typeDetails->second();
    } else {
      _treeReadBuffer = nullptr;

      if (typeName == "Int_t") {
        tree.SetBranchAddress(cleanName.c_str(), &_currentIndex);
      }
      else {
        coutE(InputArguments) << "RooAbsCategory::attachToTree(" << GetName() << ") data type " << typeName << " is not supported." << std::endl;
      }
    }
  } else {
    void* ptr = &_currentIndex;
    tree.Branch(cleanName.c_str(), ptr, (cleanName + "/I").c_str(), bufSize);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Fill tree branches associated with current object with current value

void RooAbsCategory::fillTreeBranch(TTree& t)
{
  // First determine if branch is taken
  TBranch* idxBranch = t.GetBranch((std::string(GetName()) + "_idx").c_str()) ;
  if (!idxBranch) {
    coutF(DataHandling) << "RooAbsCategory::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << std::endl;
    throw std::runtime_error("RooAbsCategory::fillTreeBranch(): Category is not attached to a tree.");
  }

  idxBranch->Fill() ;
}



////////////////////////////////////////////////////////////////////////////////
/// (De)activate associate tree branch

void RooAbsCategory::setTreeBranchStatus(TTree& t, bool active)
{
  TBranch* branch = t.GetBranch(Form("%s_idx",GetName())) ;
  if (branch) {
    t.SetBranchStatus(Form("%s_idx",GetName()),active?1:0) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Explicitly synchronize RooAbsCategory internal cache

void RooAbsCategory::syncCache(const RooArgSet*)
{
  getCurrentIndex() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy the cached value from given source and raise dirty flag.
/// It is the callers responsibility to ensure that the sources
/// cache is clean(valid) before this function is called, e.g. by
/// calling syncCache() on the source.

void RooAbsCategory::copyCache(const RooAbsArg *source, bool /*valueOnly*/, bool setValDirty)
{
   auto other = static_cast<const RooAbsCategory*>(source);
   assert(dynamic_cast<const RooAbsCategory*>(source));

   _currentIndex = other->_treeReadBuffer ? *other->_treeReadBuffer : other->_currentIndex;

   if (setValDirty) {
     setValueDirty();
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Overwrite the value stored in this object's cache.
/// This can be used to fake a computation that resulted in `value`.
/// \param[in] value Value to write. The argument is reinterpreted as a category state.
/// If such a state does not exist, this will create undefined behaviour.
/// \param[in] notifyClients If true, notify users of this object that its value changed.
/// This is the default.
void RooAbsCategory::setCachedValue(double value, bool notifyClients) {
  _currentIndex = static_cast<value_type>(value);

  if (notifyClients) {
    setValueDirty();
    _valueDirty = false;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return name and index of the `n`th defined state. When states are defined using
/// defineType() or operator[], the order of insertion is tracked, to mimic the behaviour
/// before modernising the category classes.
/// When directly manipulating the map with state names using states(), the order of insertion
/// is not known, so alphabetical ordering as usual for std::map is used. The latter is faster.
/// \param[in] n Number of state to be retrieved.
/// \return A pair with name and index.
const std::map<std::string, RooAbsCategory::value_type>::value_type& RooAbsCategory::getOrdinal(unsigned int n) const {
  // Retrieve state names, trigger possible recomputation
  auto& theStateNames = stateNames();

  if (n >= theStateNames.size())
    return invalidCategory();

  if (theStateNames.size() != _insertionOrder.size())
    return *std::next(theStateNames.begin(), n);

  const auto item = theStateNames.find(_insertionOrder[n]);
  if (item != theStateNames.end())
    return *item;

  return invalidCategory();
}


////////////////////////////////////////////////////////////////////////////////
/// Return ordinal number of the current state.
unsigned int RooAbsCategory::getCurrentOrdinalNumber() const {
  // Retrieve state names, trigger possible recomputation
  auto& theStateNames = stateNames();

  // If we don't have the full history of inserted state names, have to go by map ordering:
  if (theStateNames.size() != _insertionOrder.size()) {
    const auto currentIndex = getCurrentIndex();
    for (auto it = theStateNames.begin(); it != theStateNames.end(); ++it) {
      if (it->second == currentIndex)
        return std::distance(theStateNames.begin(), it);
    }
  }

  // With full insertion history, find index of current label:
  auto item = std::find(_insertionOrder.begin(), _insertionOrder.end(), getCurrentLabel());
  assert(item != _insertionOrder.end());

  return item - _insertionOrder.begin();
}


////////////////////////////////////////////////////////////////////////////////
/// Create a RooCategory fundamental object with our properties.

RooAbsArg *RooAbsCategory::createFundamental(const char* newname) const
{
  // Add and precalculate new category column
  RooCategory *fund= new RooCategory(newname?newname:GetName(),GetTitle()) ;

  // Copy states
  for (const auto& type : stateNames()) {
    fund->defineStateUnchecked(type.first, type.second);
  }

  return fund;
}



////////////////////////////////////////////////////////////////////////////////
/// Determine if category has 2 or 3 states with index values -1,0,1

bool RooAbsCategory::isSignType(bool mustHaveZero) const
{
  const auto& theStateNames = stateNames();

  if (theStateNames.size() > 3 || theStateNames.size() < 2) return false;
  if (mustHaveZero && theStateNames.size() != 3) return false;

  for (const auto& type : theStateNames) {
    if (std::abs(type.second)>1)
      return false;
  }

  return true;
}

/// \deprecated Use begin() and end() instead.
/// \note Using this iterator creates useless RooCatType instances, which will leak
/// unless deleted by the user.
TIterator* RooAbsCategory::typeIterator() const {
  return new RooAbsCategoryLegacyIterator(stateNames());
}

const RooCatType* RooAbsCategory::defineType(const char* label) {
  defineState(label);
  return retrieveLegacyState(stateNames()[label]);
}

const RooCatType* RooAbsCategory::defineType(const char* label, int index) {
  defineState(label, index);
  return retrieveLegacyState(index);
}

const RooCatType* RooAbsCategory::defineTypeUnchecked(const char* label, value_type index) {
  defineStateUnchecked(label, index);
  return retrieveLegacyState(index);
}

/// Return the legacy RooCatType corresponding to `index`. If it doesn't exist, create one.
RooCatType* RooAbsCategory::retrieveLegacyState(value_type index) const {
  auto result = _legacyStates.find(index);
  if (result == _legacyStates.end()) {
    result = _legacyStates.emplace(index,
        std::make_unique<RooCatType>(lookupName(index).c_str(), index)).first;
  }

  return result->second.get();
}


RooAbsCategory::value_type RooAbsCategory::nextAvailableStateIndex() const {
  const auto& theStateNames = stateNames();

  if (theStateNames.empty())
    return 0;

  return 1 + std::max_element(theStateNames.begin(), theStateNames.end(),
      [](auto const& left, auto const& right) { return left.second < right.second; })->second;
}
