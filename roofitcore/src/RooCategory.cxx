/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooCategory.cc,v 1.30 2006/07/03 15:37:11 wverkerke Exp $
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

// -- CLASS DESCRIPTION [CAT] --
// RooCategory represents a fundamental (non-derived) discrete value object. The class
// has a public interface to define the possible value states.


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <stdlib.h>
#include <string.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooCategory.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"

ClassImp(RooCategory) 
;

RooSharedPropertiesList RooCategory::_sharedPropList ;


RooCategory::RooCategory()
{
}

RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategoryLValue(name,title)
{
  // Constructor. Types must be defined using defineType() before variable can be used

  _sharedProp = (RooCategorySharedProperties*) _sharedPropList.registerProperties(new RooCategorySharedProperties()) ;

  setValueDirty() ;  
  setShapeDirty() ;  
}


RooCategory::RooCategory(const RooCategory& other, const char* name) :
  RooAbsCategoryLValue(other, name)
{
  // Copy constructor
  _sharedProp =  (RooCategorySharedProperties*) _sharedPropList.registerProperties(other._sharedProp) ;
  
}


RooCategory::~RooCategory()
{
  // Destructor
  _sharedPropList.unregisterProperties(_sharedProp) ;
}



Bool_t RooCategory::setIndex(Int_t index, Bool_t printError) 
{
  // Set value by specifying the index code of the desired state.
  // If printError is set, a message will be printed if
  // the specified index does not represent a valid state.

  const RooCatType* type = lookupType(index,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



Bool_t RooCategory::setLabel(const char* label, Bool_t printError) 
{
  // Set value by specifying the name of the desired state
  // If printError is set, a message will be printed if
  // the specified label does not represent a valid state.

  const RooCatType* type = lookupType(label,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



Bool_t RooCategory::defineType(const char* label) 
{ 
  // Define a state with given name, the lowest available
  // positive integer is assigned as index. Category
  // state labels may not contain semicolons.
  // Error status is return if state with given name
  // is already defined

  if (TString(label).Contains(";")) {
  cout << "RooCategory::defineType(" << GetName() 
       << "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label)?kFALSE:kTRUE ; 
}


Bool_t RooCategory::defineType(const char* label, Int_t index) 
{
  // Define a state with given name and index. Category
  // state labels may not contain semicolons
  // Error status is return if state with given name
  // or index is already defined

  if (TString(label).Contains(";")) {
  cout << "RooCategory::defineType(" << GetName() 
       << "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label,index)?kFALSE:kTRUE ; 
}


Bool_t RooCategory::readFromStream(istream& is, Bool_t /*compact*/, Bool_t verbose) 
{
  // Read object contents from given stream

  // Read single token
  RooStreamParser parser(is) ;
  TString token = parser.readToken() ;

  return setLabel(token,verbose) ;
}



void RooCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // compact only at the moment
  if (compact) {
    os << getIndex() ;
  } else {
    os << getLabel() ;
  }
}


void RooCategory::clearRange(const char* name, Bool_t silent)
{
  // Check that both input arguments are not null pointers
  if (!name) {
    cout << "RooCategory::clearRange(" << GetName() << ") ERROR: must specificy valid range name" << endl ;
    return ;
  }
  
  // Find the list that represents this range
  TList* rangeNameList = static_cast<TList*>(_sharedProp->_altRanges.FindObject(name)) ;

  // If it exists, clear it 
  if (rangeNameList) {
    rangeNameList->Clear() ;
  } else if (!silent) {
    cout << "RooCategory::clearRange(" << GetName() << ") ERROR: range '" << name << "' does not exist" << endl ;
  } 
}


void RooCategory::setRange(const char* name, const char* stateNameList) 
{
  clearRange(name,kTRUE) ;
  addToRange(name,stateNameList) ;
}


void RooCategory::addToRange(const char* name, const char* stateNameList) 
{
  // Check that both input arguments are not null pointers
  if (!name || !stateNameList) {
    cout << "RooCategory::setRange(" << GetName() << ") ERROR: must specificy valid name and state name list" << endl ;
    return ;
  }
  
  // Find the list that represents this range
  TList* rangeNameList = static_cast<TList*>(_sharedProp->_altRanges.FindObject(name)) ;

  // If it does not exist, create it on the fly
  if (!rangeNameList) {
    cout << "RooCategory::setRange(" << GetName() 
	 << ") new range named '" << name << "' created with state list " << stateNameList << endl ;

    rangeNameList = new TList ;
    rangeNameList->SetName(name) ;
    _sharedProp->_altRanges.Add(rangeNameList) ;    
  }

  // Parse list of state names, verify that each is valid and add them to the list
  char* buf = new char[strlen(stateNameList)+1] ;
  strcpy(buf,stateNameList) ;
  char* token = strtok(buf,",") ;
  while(token) {
    const RooCatType* state = lookupType(token,kFALSE) ;
    if (state && !rangeNameList->FindObject(token)) {
      rangeNameList->Add(const_cast<RooCatType*>(state)) ;	
    } else {
      cout << "RooCategory::setRange(" << GetName() << ") WARNING: Ignoring invalid state name '" 
	   << token << "' in state name list" << endl ;
    }
    token = strtok(0,",") ;
  }

  delete[] buf ;
}

Bool_t RooCategory::isStateInRange(const char* rangeName, const char* stateName) const
{
  // Check that both input arguments are not null pointers
  if (!rangeName||!stateName) {
    cout << "RooCategory::isStateInRange(" << GetName() << ") ERROR: must specificy valid range name and state name" << endl ;
    return kFALSE ;
  }

  
  // Find the list that represents this range
  TList* rangeNameList = static_cast<TList*>(_sharedProp->_altRanges.FindObject(rangeName)) ;

  // If the range doesn't exist create range with all valid states included
  if (rangeNameList) {
    return rangeNameList->FindObject(stateName) ? kTRUE : kFALSE ;  
  }

  // Range does not exists -- create it on the fly with full set of states (analoguous to RooRealVar)
  return kTRUE ;

}


