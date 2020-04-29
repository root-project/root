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
\file RooCategory.cxx
\class RooCategory
\ingroup Roofitcore

RooCategory represents a fundamental (non-derived) discrete category object. "Fundamental" means that
it can be written into a dataset. (Objects in datasets cannot depend on other objects' values,
they need to have their own value). A category object can be used to *e.g.* conduct a simultaneous fit of
the same observable in multiple categories
The states of the category can be denoted by integers (faster) or state names.

A category can be set up like this:
~~~{.cpp}
RooCategory myCat("myCat", "Lepton multiplicity category", {
                  {"0Lep", 0},
                  {"1Lep", 1},
                  {"2Lep", 2},
                  {"3Lep", 3}
});
~~~
Or like this:
~~~{.cpp}
RooCategory myCat("myCat", "Asymmetry");
myCat.defineType("left", -1);
myCat.defineType("right", 1);
~~~
Inspect the pairs of index number and state names like this:
~~~{.cpp}
for (const auto& idxAndName : myCat) {
  std::cout << idxAndName.first << " --> " << idxAndName.second << std::endl;
}
~~~

Also refer to the RooFit tutorials rf404_categories.C for an introduction, and to rf405_realtocatfuncs.C and rf406_cattocatfuncs.C
for advanced uses of categories.
**/

#include "RooCategory.h"

#include "RooFit.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooTrace.h"
#include "RooHelpers.h"
#include "RooCategorySharedProperties.h"
#include "RooFitLegacy/RooCatTypeLegacy.h"

#include "TBuffer.h"
#include "TString.h"

using namespace std;

ClassImp(RooCategory); 


////////////////////////////////////////////////////////////////////////////////

RooCategory::RooCategory()
{
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor. Types must be defined using defineType() before variable can be used
RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategoryLValue(name,title),
  _ranges(new RangeMap_t())
{
  setValueDirty() ;  
  setShapeDirty() ;  
  TRACE_CREATE 
}


////////////////////////////////////////////////////////////////////////////////
/// Create a new category and define allowed states.
/// \param[in] name Name used to refer to this object.
/// \param[in] title Title for e.g. plotting.
/// \param[in] allowedStates Map of allowed states. Pass e.g. `{ {"0Lep", 0}, {"1Lep:, 1} }`
RooCategory::RooCategory(const char* name, const char* title, const std::map<std::string, int>& allowedStates) :
  RooAbsCategoryLValue(name,title),
  _ranges(new RangeMap_t())
{
  defineTypes(allowedStates);
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCategory::RooCategory(const RooCategory& other, const char* name) :
  RooAbsCategoryLValue(other, name),
  _ranges(other._ranges)
{
  TRACE_CREATE   
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooCategory::~RooCategory()
{
  TRACE_DESTROY
}




////////////////////////////////////////////////////////////////////////////////
/// Set value by specifying the index code of the desired state.
/// If printError is set, a message will be printed if
/// the specified index does not represent a valid state.
/// \return bool signalling if an error occurred.
Bool_t RooCategory::setIndex(Int_t index, Bool_t printError) 
{
  if (!hasIndex(index)) {
    if (printError) {
      coutE(InputArguments) << "RooCategory: Trying to set invalid state " << index << " for category " << GetName() << std::endl;
    }
    return true;
  }

  _currentIndex = index;
  setValueDirty();

  return false;
}



////////////////////////////////////////////////////////////////////////////////
/// Set value by specifying the name of the desired state.
/// If printError is set, a message will be printed if
/// the specified label does not represent a valid state.
/// \return false on success.
Bool_t RooCategory::setLabel(const char* label, Bool_t printError) 
{
  const auto item = stateNames().find(label);
  if (item != stateNames().end()) {
    _currentIndex = item->second;
    setValueDirty();
    return false;
  }

  if (printError) {
    coutE(InputArguments) << "Trying to set invalid state label '" << label << "' for category " << GetName() << std::endl;
  }

  return true;
}



////////////////////////////////////////////////////////////////////////////////
/// Define a state with given name.
/// The lowest available positive integer is assigned as index. Category
/// state labels may not contain semicolons.
/// \return True in case of an error.
bool RooCategory::defineType(const std::string& label)
{ 
  if (label.find(';') != std::string::npos) {
    coutE(InputArguments) << "RooCategory::defineType(" << GetName()
        << "): semicolons not allowed in label name" << endl ;
    return true;
  }

  return RooAbsCategory::defineState(label) == RooAbsCategory::_invalidCategory;
}


////////////////////////////////////////////////////////////////////////////////
/// Define a state with given name and index. Category
/// state labels may not contain semicolons.
/// \return True in case of error.
bool RooCategory::defineType(const std::string& label, Int_t index)
{
  if (label.find(';') != std::string::npos) {
    coutE(InputArguments) << "RooCategory::defineType(" << GetName()
			<< "): semicolons not allowed in label name" << endl ;
    return true;
  }

  return RooAbsCategory::defineState(label, index) == RooAbsCategory::_invalidCategory;
}


////////////////////////////////////////////////////////////////////////////////
/// Define multiple states in a single call. Use like:
/// ```
/// myCat.defineTypes({ {"0Lep", 0}, {"1Lep", 1}, {"2Lep", 2}, {"3Lep", 3} });
/// ```
/// Note: When labels or indices are defined multiple times, an error message is printed,
/// and the corresponding state is ignored.
void RooCategory::defineTypes(const std::map<std::string, int>& allowedStates) {
  for (const auto& nameAndIdx : allowedStates) {
    defineType(nameAndIdx.first, nameAndIdx.second);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Access a named state. If a state with this name doesn't exist yet, the state is
/// assigned the next available positive integer.
/// \param[in] stateName Name of the state to be accessed.
/// \return Reference to the category index. If no state exists, it will be created on the fly.
RooAbsCategory::value_type& RooCategory::operator[](const std::string& stateName) {
  setShapeDirty();
  if (stateNames().count(stateName) == 0)
    return stateNames()[stateName] = nextAvailableStateIndex();

  return stateNames()[stateName];
}


////////////////////////////////////////////////////////////////////////////////
/// Return a reference to the map of state names to index states.
/// This can be used to manipulate the category.
/// \note Calling this function will **always** trigger recomputations of
/// of **everything** that depends on this category, since in case the map gets
/// manipulated, names or indices might change.
std::map<std::string, RooAbsCategory::value_type>& RooCategory::states() {
  auto& theStates = stateNames();
  setValueDirty();
  setShapeDirty();
  return theStates;
}


////////////////////////////////////////////////////////////////////////////////
/// Read object contents from given stream

Bool_t RooCategory::readFromStream(istream& is, Bool_t /*compact*/, Bool_t verbose) 
{
  // Read single token
  RooStreamParser parser(is) ;
  TString token = parser.readToken() ;

  return setLabel(token,verbose) ;
}



////////////////////////////////////////////////////////////////////////////////
/// compact only at the moment

void RooCategory::writeToStream(ostream& os, Bool_t compact) const
{
  if (compact) {
    os << getIndex() ;
  } else {
    os << getLabel() ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Clear the named range.
/// \note This affects **all** copies of this category, because they are sharing
/// range definitions. This ensures that categories inside a dataset and their
/// counterparts on the outside will both see a modification of the range.
void RooCategory::clearRange(const char* name, Bool_t silent)
{
  std::map<std::string, std::vector<value_type>>::iterator item = _ranges->find(name);
  if (item == _ranges->end()) {
    if (!silent)
      coutE(InputArguments) << "RooCategory::clearRange(" << GetName() << ") ERROR: must specify valid range name" << endl ;
    return;
  }
  
  _ranges->erase(item);
}


////////////////////////////////////////////////////////////////////////////////

void RooCategory::setRange(const char* name, const char* stateNameList) 
{
  clearRange(name,kTRUE) ;
  addToRange(name,stateNameList) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Add the given state to the given range.
/// \note This creates or accesses a **shared** map with allowed ranges. All copies of this
/// category will share this range such that a category inside a dataset and its
/// counterpart on the outside will both see a modification of the range.
void RooCategory::addToRange(const char* name, RooAbsCategory::value_type stateIndex) {
  auto item = _ranges->find(name);
  if (item == _ranges->end()) {
    if (!name) {
      coutE(Contents) << "RooCategory::addToRange(" << GetName()
          << "): Need valid range name." << std::endl;
      return;
    }

    item = _ranges->emplace(name, std::vector<value_type>()).first;
    coutI(Contents) << "RooCategory::setRange(" << GetName() 
        << ") new range named '" << name << "' created for state " << stateIndex << endl ;
  }

  item->second.push_back(stateIndex);
}


////////////////////////////////////////////////////////////////////////////////
/// Add the list of state names to the given range. State names can be separated
/// with ','.
/// \note This creates or accesses a **shared** map with allowed ranges. All copies of this
/// category will share this range such that a category inside a dataset and its
/// counterpart on the outside will both see a modification of the range.
void RooCategory::addToRange(const char* name, const char* stateNameList)
{
  if (!stateNameList) {
    coutE(InputArguments) << "RooCategory::setRange(" << GetName() << ") ERROR: must specify valid name and state name list" << endl ;
    return;
  }

  // Parse list of state names, verify that each is valid and add them to the list
  for (const auto& token : RooHelpers::tokenise(stateNameList, ",")) {
    const value_type idx = lookupIndex(token);
    if (idx != _invalidCategory.second) {
      addToRange(name, idx);
    } else {
      coutW(InputArguments) << "RooCategory::setRange(" << GetName() << ") WARNING: Ignoring invalid state name '" 
			    << token << "' in state name list" << endl ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Check if the state is in the given range.
/// If no range is specified either as argument or if no range has been defined for this category
/// (*i.e.*, the default range is meant), all category states count as being in range.
bool RooCategory::isStateInRange(const char* rangeName, RooAbsCategory::value_type stateIndex) const {
  if (rangeName == nullptr || _ranges->empty())
    return true;

  const auto item = _ranges->find(rangeName);
  if (item == _ranges->end())
    return false;

  const std::vector<value_type>& vec = item->second;
  return std::find(vec.begin(), vec.end(), stateIndex) != vec.end();
}


////////////////////////////////////////////////////////////////////////////////
/// Check if the state is in the given range.
/// If no range is specified (*i.e.*, the default range), all category states count as being in range.
/// This overload requires a name lookup. Recommend to use the category index with
/// RooCategory::isStateInRange(const char*, RooAbsCategory::value_type) const.
bool RooCategory::isStateInRange(const char* rangeName, const char* stateName) const
{
  // Check that both input arguments are not null pointers
  if (!rangeName) {
    return true;
  }

  if (!stateName) {
    coutE(InputArguments) << "RooCategory::isStateInRange(" << GetName() << ") ERROR: must specify valid state name" << endl ;
    return false;
  }

  return isStateInRange(rangeName, lookupIndex(stateName));
}


////////////////////////////////////////////////////////////////////////////////

void RooCategory::Streamer(TBuffer &R__b)
{
  UInt_t R__s, R__c;
  if (R__b.IsReading()) {
    
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

    if (R__v > 2) {
      // Before version 3, ranges were shared using RooCategorySharedProperties.
      // Now, it is a shared pointer, which cannot be read by ROOT's I/O. Instead,
      // a normal pointer is read, and later assigned to the shared pointer. Like this
      // clones of this category will share the same ranges.
      R__b.ReadClassBuffer(RooCategory::Class(), this, R__v, R__s, R__c);
      if (_rangesPointerForIO) {
        _ranges.reset(_rangesPointerForIO);
        _rangesPointerForIO = nullptr;
      }
    } else {

      RooAbsCategoryLValue::Streamer(R__b);

      RooCategorySharedProperties* props = nullptr;
      if (R__v==1) {
        // Implement V1 streamer here
        R__b >> props;
      } else {
        props = new RooCategorySharedProperties();
        props->Streamer(R__b);
        if (*props == RooCategorySharedProperties("00000000-0000-0000-0000-000000000000")) {
          delete props;
          props = nullptr;
        }
      }

      if (props) {
        _ranges.reset(new std::map<std::string, std::vector<value_type>>());
        auto& rangesMap = *_ranges;
        std::unique_ptr<TIterator> iter(props->_altRanges.MakeIterator());
        TList* olist;
        while((olist=(TList*)iter->Next())) {
          std::vector<value_type>& vec = rangesMap[olist->GetName()];

          RooCatType* ctype ;
          std::unique_ptr<TIterator> citer(olist->MakeIterator());
          while ((ctype=(RooCatType*)citer->Next())) {
            vec.push_back(ctype->getVal());
          }
        }
      }
      delete props;
    }

    R__b.CheckByteCount(R__s, R__c, RooCategory::IsA());
    
  } else {
    // Since we cannot write shared pointers yet, assign the shared ranges to a normal pointer
    // while we are writing.
    if (_ranges)
      _rangesPointerForIO = _ranges.get();

    R__b.WriteClassBuffer(RooCategory::Class(), this);
    _rangesPointerForIO = nullptr;
  }
}
