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
// RooChangeTracker is a meta object that tracks value
// changes in a given set of RooAbsArgs by registering itself as value
// client of these objects. The change tracker can perform an
// additional validation step where it also compares the numeric
// values of the tracked arguments with reference values to ensure
// that values have actually changed. This may be useful in case some
// of the tracked observables are in binned datasets where each
// observable propates a valueDirty flag when an event is loaded even
// though usually only one observable actually changes.
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include <math.h>

#include "RooChangeTracker.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgSet.h"
#include "RooMsgService.h"

using namespace std ;

ClassImp(RooChangeTracker)
;

//_____________________________________________________________________________
RooChangeTracker::RooChangeTracker() : _checkVal(kFALSE), _init(kFALSE)
{
  // Default constructor

  _realSetIter = _realSet.createIterator() ;
  _catSetIter = _catSet.createIterator() ;
}



//_____________________________________________________________________________
RooChangeTracker::RooChangeTracker(const char* name, const char* title, const RooArgSet& trackSet, Bool_t checkValues) :
  RooAbsReal(name, title),
  _realSet("realSet","Set of real-valued components to be tracked",this),
  _catSet("catSet","Set of discrete-valued components to be tracked",this),
  _realRef(trackSet.getSize()),
  _catRef(trackSet.getSize()),
  _checkVal(checkValues),
  _init(kFALSE)
{
  // Constructor. The set trackSet contains the observables to be
  // tracked for changes. If checkValues is true an additional
  // validation step is activated where the numeric values of the
  // tracked arguments are compared with reference values ensuring
  // that values have actually changed.

  _realSetIter = _realSet.createIterator() ;
  _catSetIter = _catSet.createIterator() ;

  TIterator* iter = trackSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (dynamic_cast<RooAbsReal*>(arg)) {
      _realSet.add(*arg) ;      
    }
    if (dynamic_cast<RooAbsCategory*>(arg)) {
      _catSet.add(*arg) ;      
    }
  }
  delete iter ;
  
  if (_checkVal) {
    RooAbsReal* real  ;
    RooAbsCategory* cat  ;
    Int_t i(0) ;
    _realSetIter->Reset() ;
    _catSetIter->Reset() ;
    while((real=(RooAbsReal*)_realSetIter->Next())) {
      _realRef[i++] = real->getVal() ;
    }
    i=0 ;
    while((cat=(RooAbsCategory*)_catSetIter->Next())) {
      _catRef[i++] = cat->getIndex() ;
    }
  }

}



//_____________________________________________________________________________
RooChangeTracker::RooChangeTracker(const RooChangeTracker& other, const char* name) :
  RooAbsReal(other, name), 
  _realSet("realSet",this,other._realSet),
  _catSet("catSet",this,other._catSet),
  _realRef(other._realRef),
  _catRef(other._catRef),
  _checkVal(other._checkVal),
  _init(kFALSE)
{
  // Copy constructor

  _realSetIter = _realSet.createIterator() ;
  _catSetIter = _catSet.createIterator() ;

//   _realSet.add(other._realSet) ;
//   _catSet.add(other._catSet) ;

}



//_____________________________________________________________________________
Bool_t RooChangeTracker::hasChanged(Bool_t clearState) 
{
  // Returns true if state has changes since last call with clearState=kTRUE
  // If clearState is true, changeState flag will be cleared.


  // If dirty flag did not change, object has not changed in any case
  if (!isValueDirty()) {
    return kFALSE ;
  }

  // If no value checking is required and dirty flag has changed, return true
  if (!_checkVal) {

    if (clearState) {
      // Clear dirty flag by calling getVal()
      //cout << "RooChangeTracker(" << GetName() << ") clearing isValueDirty" << endl ;
      clearValueDirty() ;
    }

    //cout << "RooChangeTracker(" << GetName() << ") isValueDirty = kTRUE, returning kTRUE" << endl ;

    return kTRUE ;
  }
  
  // Compare values against reference
  _realSetIter->Reset() ;  
  _catSetIter->Reset() ;
  RooAbsReal* real ;
  RooAbsCategory* cat ;
  Int_t i(0) ;

  if (clearState) {

    Bool_t valuesChanged(kFALSE) ;

    // Check if any of the real values changed
    while ((real=(RooAbsReal*)_realSetIter->Next())) {
      if (real->getVal() != _realRef[i]) {
	// cout << "RooChangeTracker(" << this << "," << GetName() << ") value of " << real->GetName() << " has changed from " << _realRef[i] << " to " << real->getVal() << " clearState = " << (clearState?"T":"F") << endl ;
	valuesChanged = kTRUE ;
	_realRef[i] = real->getVal() ;
      }
      i++ ;
    }
    // Check if any of the categories changed
    i=0 ;
    while ((cat=(RooAbsCategory*)_catSetIter->Next())) {
      if (cat->getIndex() != _catRef[i++]) {
	// cout << "RooChangeTracker(" << this << "," << GetName() << ") value of " << cat->GetName() << " has changed from " << _catRef[i-1] << " to " << cat->getIndex() << endl ;
	valuesChanged = kTRUE ;
	_catRef[i-1] = cat->getIndex() ;
      }
    }

    clearValueDirty() ;


    if (!_init) {
      valuesChanged=kTRUE ;
      _init = kTRUE ;
    }
    
    // cout << "RooChangeTracker(" << GetName() << ") returning " << (valuesChanged?"T":"F") << endl ;

    return valuesChanged ;

  } else {
    
    // Return true as soon as any input has changed

    // Check if any of the real values changed
    while ((real=(RooAbsReal*)_realSetIter->Next())) {
      if (real->getVal() != _realRef[i++]) {
	return kTRUE ;
      }
    }
    // Check if any of the categories changed
    i=0 ;
    while ((cat=(RooAbsCategory*)_catSetIter->Next())) {
      if (cat->getIndex() != _catRef[i++]) {
	return kTRUE ;
      }
    }
   
  }
  
  return kFALSE ;
}



//_____________________________________________________________________________
RooChangeTracker::~RooChangeTracker() 
{
  // Destructor
  if (_realSetIter) delete _realSetIter ;
  if (_catSetIter) delete _catSetIter ;
}



//_____________________________________________________________________________
RooArgSet RooChangeTracker::parameters() const 
{
  RooArgSet ret ;
  ret.add(_realSet) ;
  ret.add(_catSet) ;
  return ret ;
}



