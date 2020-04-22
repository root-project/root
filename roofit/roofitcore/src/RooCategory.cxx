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

RooCategory represents a fundamental (non-derived) discrete value object. The class
has a public interface to define the possible value states.
**/


#include "RooFit.h"

#include "Riostream.h"
#include <stdlib.h>
#include "TTree.h"
#include "TString.h"
#include "TH1.h"
#include "RooCategory.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"
#include "RooTrace.h"
#include "TBuffer.h"

using namespace std;

ClassImp(RooCategory); 


RooSharedPropertiesList RooCategory::_sharedPropList ;
RooCategorySharedProperties RooCategory::_nullProp("00000000-0000-0000-0000-000000000000") ;

////////////////////////////////////////////////////////////////////////////////

RooCategory::RooCategory() : _sharedProp(0)
{
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor. Types must be defined using defineType() before variable can be used

RooCategory::RooCategory(const char *name, const char *title) : 
  RooAbsCategoryLValue(name,title)
{
  _sharedProp = (RooCategorySharedProperties*) _sharedPropList.registerProperties(new RooCategorySharedProperties()) ;

  setValueDirty() ;  
  setShapeDirty() ;  
  TRACE_CREATE 
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooCategory::RooCategory(const RooCategory& other, const char* name) :
  RooAbsCategoryLValue(other, name)
{
  _sharedProp =  (RooCategorySharedProperties*) _sharedPropList.registerProperties(other._sharedProp) ;
  TRACE_CREATE   
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooCategory::~RooCategory()
{
  _sharedPropList.unregisterProperties(_sharedProp) ;
  TRACE_DESTROY
}




////////////////////////////////////////////////////////////////////////////////
/// Set value by specifying the index code of the desired state.
/// If printError is set, a message will be printed if
/// the specified index does not represent a valid state.

Bool_t RooCategory::setIndex(Int_t index, Bool_t printError) 
{
  const RooCatType* type = lookupType(index,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set value by specifying the name of the desired state
/// If printError is set, a message will be printed if
/// the specified label does not represent a valid state.

Bool_t RooCategory::setLabel(const char* label, Bool_t printError) 
{
  const RooCatType* type = lookupType(label,printError) ;
  if (!type) return kTRUE ;
  _value = *type ;
  setValueDirty() ;
  return kFALSE ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define a state with given name, the lowest available
/// positive integer is assigned as index. Category
/// state labels may not contain semicolons.
/// Error status is return if state with given name
/// is already defined

Bool_t RooCategory::defineType(const char* label) 
{ 
  if (TString(label).Contains(";")) {
  coutE(InputArguments) << "RooCategory::defineType(" << GetName() 
			<< "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label)?kFALSE:kTRUE ; 
}


////////////////////////////////////////////////////////////////////////////////
/// Define a state with given name and index. Category
/// state labels may not contain semicolons
/// Error status is return if state with given name
/// or index is already defined

Bool_t RooCategory::defineType(const char* label, Int_t index) 
{
  if (TString(label).Contains(";")) {
  coutE(InputArguments) << "RooCategory::defineType(" << GetName() 
			<< "): semicolons not allowed in label name" << endl ;
  return kTRUE ;
  }

  return RooAbsCategory::defineType(label,index)?kFALSE:kTRUE ; 
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
/// Check that both input arguments are not null pointers

void RooCategory::clearRange(const char* name, Bool_t silent)
{
  if (!name) {
    coutE(InputArguments) << "RooCategory::clearRange(" << GetName() << ") ERROR: must specificy valid range name" << endl ;
    return ;
  }
  
  // Find the list that represents this range
  TList* rangeNameList = static_cast<TList*>(_sharedProp->_altRanges.FindObject(name)) ;

  // If it exists, clear it 
  if (rangeNameList) {
    rangeNameList->Clear() ;
  } else if (!silent) {
    coutE(InputArguments) << "RooCategory::clearRange(" << GetName() << ") ERROR: range '" << name << "' does not exist" << endl ;
  } 
}


////////////////////////////////////////////////////////////////////////////////

void RooCategory::setRange(const char* name, const char* stateNameList) 
{
  clearRange(name,kTRUE) ;
  addToRange(name,stateNameList) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check that both input arguments are not null pointers

void RooCategory::addToRange(const char* name, const char* stateNameList) 
{
  if (!name || !stateNameList) {
    coutE(InputArguments) << "RooCategory::setRange(" << GetName() << ") ERROR: must specificy valid name and state name list" << endl ;
    return ;
  }
  
  // Find the list that represents this range
  TList* rangeNameList = static_cast<TList*>(_sharedProp->_altRanges.FindObject(name)) ;

  // If it does not exist, create it on the fly
  if (!rangeNameList) {
    coutI(Contents) << "RooCategory::setRange(" << GetName() 
		    << ") new range named '" << name << "' created with state list " << stateNameList << endl ;

    rangeNameList = new TList ;
    rangeNameList->SetOwner(kTRUE) ;
    rangeNameList->SetName(name) ;
    _sharedProp->_altRanges.Add(rangeNameList) ;    
  }

  // Parse list of state names, verify that each is valid and add them to the list
  const size_t bufSize = strlen(stateNameList)+1;
  char* buf = new char[bufSize] ;
  strlcpy(buf,stateNameList,bufSize) ;
  char* token = strtok(buf,",") ;
  while(token) {
    const RooCatType* state = lookupType(token,kFALSE) ;
    if (state && !rangeNameList->FindObject(token)) {
      rangeNameList->Add(new RooCatType(*state)) ;	
    } else {
      coutW(InputArguments) << "RooCategory::setRange(" << GetName() << ") WARNING: Ignoring invalid state name '" 
			    << token << "' in state name list" << endl ;
    }
    token = strtok(0,",") ;
  }

  delete[] buf ;
}



////////////////////////////////////////////////////////////////////////////////
/// If no range is specified [ i.e. the default range ] all category states are in range

Bool_t RooCategory::isStateInRange(const char* rangeName, const char* stateName) const
{
  if (!rangeName) {
    return kTRUE ;
  }

  // Check that both input arguments are not null pointers
  if (!stateName) {
    coutE(InputArguments) << "RooCategory::isStateInRange(" << GetName() << ") ERROR: must specificy valid state name" << endl ;
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


////////////////////////////////////////////////////////////////////////////////

void RooCategory::Streamer(TBuffer &R__b)
{
  UInt_t R__s, R__c;
  if (R__b.IsReading()) {
    
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }    
    RooAbsCategoryLValue::Streamer(R__b);
    if (R__v==1) {
      // Implement V1 streamer here
      R__b >> _sharedProp;      
    } else { 
      RooCategorySharedProperties* tmpSharedProp = new RooCategorySharedProperties() ;
      tmpSharedProp->Streamer(R__b) ;
      if (!(_nullProp==*tmpSharedProp)) {
	_sharedProp = (RooCategorySharedProperties*) _sharedPropList.registerProperties(tmpSharedProp,kFALSE) ;
      } else {
	delete tmpSharedProp ;
	_sharedProp = 0 ;
      }
    }

    R__b.CheckByteCount(R__s, R__c, RooCategory::IsA());
    
  } else {
    
    R__c = R__b.WriteVersion(RooCategory::IsA(), kTRUE);
    RooAbsCategoryLValue::Streamer(R__b);
    if (_sharedProp) {
      _sharedProp->Streamer(R__b) ;
    } else {
      _nullProp.Streamer(R__b) ;
    }
    R__b.SetByteCount(R__c, kTRUE);      
    
  }
}


