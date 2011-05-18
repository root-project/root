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
// RooAICRegistry is a utility class for operator p.d.f
// classes that keeps track of analytical integration codes and
// associated normalization and integration sets.  
// END_HTML
//

#include "RooFit.h"

#include "RooAICRegistry.h"
#include "RooMsgService.h"
#include "RooArgSet.h"
#include "RooMsgService.h"

#include "Riostream.h"


ClassImp(RooAICRegistry)
;


//_____________________________________________________________________________
RooAICRegistry::RooAICRegistry(Int_t regSize) :
  _regSize(regSize), _clSize(0), _clArr(0), _asArr1(0), _asArr2(0), _asArr3(0), _asArr4(0)
{
  // Constructor
}



//_____________________________________________________________________________
RooAICRegistry::RooAICRegistry(const RooAICRegistry& other) :
  _regSize(other._regSize), _clSize(0), _clArr(0), _asArr1(0), _asArr2(0), _asArr3(0), _asArr4(0)
{
  // Copy constructor

  // Copy code-list array if other PDF has one
  if (other._clArr) {
    _clArr = new pInt_t[other._regSize] ;    
    _asArr1 = new pRooArgSet[other._regSize] ;
    _asArr2 = new pRooArgSet[other._regSize] ;
    _asArr3 = new pRooArgSet[other._regSize] ;
    _asArr4 = new pRooArgSet[other._regSize] ;
    _clSize = new Int_t[other._regSize] ;
    Int_t i,j ;
    for (i=0 ; i<_regSize ; i++) {
      _clArr[i]=0 ;
      _clSize[i]=0 ;
      _asArr1[i]=0 ;
      _asArr2[i]=0 ;
      _asArr3[i]=0 ;
      _asArr4[i]=0 ;
    }
    i=0 ;
    while(other._clArr[i] && i<_regSize) {
      _clSize[i] = other._clSize[i] ;
      _asArr1[i] = other._asArr1[i] ? ((RooArgSet*)other._asArr1[i]->snapshot(kFALSE)) : 0 ; 
      _asArr2[i] = other._asArr2[i] ? ((RooArgSet*)other._asArr2[i]->snapshot(kFALSE)) : 0 ;
      _asArr3[i] = other._asArr3[i] ? ((RooArgSet*)other._asArr3[i]->snapshot(kFALSE)) : 0 ;
      _asArr4[i] = other._asArr4[i] ? ((RooArgSet*)other._asArr4[i]->snapshot(kFALSE)) : 0 ;
      _clArr[i] = new Int_t[_clSize[i]] ;
      for (j=0 ; j<_clSize[i] ; j++) {
	_clArr[i][j] = other._clArr[i][j] ;
      }
      i++ ;
    }
  }
}



//_____________________________________________________________________________
RooAICRegistry::~RooAICRegistry() 
{
  // Destructor

  // Delete code list array, if allocated
  if (_clArr) {
    Int_t i(0) ;
    while(_clArr[i] && i<_regSize) {
      delete[] _clArr[i] ;
      if (_asArr1[i]) delete _asArr1[i] ;
      if (_asArr2[i]) delete _asArr2[i] ;
      if (_asArr3[i]) delete _asArr3[i] ;
      if (_asArr4[i]) delete _asArr4[i] ;
      i++ ;
    }
    delete[] _clArr ;
    delete[] _clSize ;
    delete[] _asArr1 ;
    delete[] _asArr2 ;
    delete[] _asArr3 ;
    delete[] _asArr4 ;
  }
}



//_____________________________________________________________________________
Int_t RooAICRegistry::store(Int_t* codeList, Int_t size, RooArgSet* set1, RooArgSet* set2, RooArgSet* set3, RooArgSet* set4)
{
  // Store given arrays of integer codes, and up to four RooArgSets in
  // the registry (each setX pointer may be null). The size of the
  // arrays should be passed by the 'size' argument. The registry
  // clones all RooArgSets internally so the RooArgSets passed as
  // arguments do not need to live beyond the store() call. The return
  // value is a unique master code for the given configuration of
  // integers and RooArgSets. If an identical combination is
  // previously stored in the registry no objects are stored and the
  // unique code of the existing entry is returned.

  Int_t i,j ;

  // If code list array has never been used, allocate and initialize here
  if (!_clArr) {
    _clArr = new pInt_t[_regSize] ;
    _clSize = new Int_t[_regSize] ;
    _asArr1 = new pRooArgSet[_regSize] ;
    _asArr2 = new pRooArgSet[_regSize] ;
    _asArr3 = new pRooArgSet[_regSize] ;
    _asArr4 = new pRooArgSet[_regSize] ;
    for (i=0 ; i<_regSize ; i++) {
      _clArr[i] = 0 ;    
      _clSize[i] = 0 ;
      _asArr1[i] = 0 ;
      _asArr2[i] = 0 ;
      _asArr3[i] = 0 ;
      _asArr4[i] = 0 ;
    }
  }

  // Loop over code-list array
  for (i=0 ; i<_regSize ; i++) {
    if (_clArr[i]==0) {
      // Empty slot, store code list and return index
      _clArr[i] = new Int_t[size] ;
      _clSize[i] = size ;
      _asArr1[i] = set1 ? (RooArgSet*)set1->snapshot(kFALSE) : 0;
      _asArr2[i] = set2 ? (RooArgSet*)set2->snapshot(kFALSE) : 0;
      _asArr3[i] = set3 ? (RooArgSet*)set3->snapshot(kFALSE) : 0;
      _asArr4[i] = set4 ? (RooArgSet*)set4->snapshot(kFALSE) : 0;
      for (j=0 ; j<size ; j++) _clArr[i][j] = codeList[j] ;

      if (set1) delete set1 ;
      if (set2) delete set2 ;
      if (set3) delete set3 ;
      if (set4) delete set4 ;

      return i ;
    } else {

      // Existing slot, compare with current list, if matched return index
      Bool_t match(kTRUE) ;

      // Check that array contents is identical
      for (j=0 ; j<size ; j++) {
	if (_clArr[i][j] != codeList[j]) match=kFALSE ;
      }
      // Check that supplied configuration of lists is identical
      if (_asArr1[i] && !set1) match=kFALSE ;
      if (!_asArr1[i] && set1) match=kFALSE ;
      if (_asArr2[i] && !set2) match=kFALSE ;
      if (!_asArr2[i] && set2) match=kFALSE ;
      if (_asArr3[i] && !set3) match=kFALSE ;
      if (!_asArr3[i] && set3) match=kFALSE ;
      if (_asArr4[i] && !set4) match=kFALSE ;
      if (!_asArr4[i] && set4) match=kFALSE ;

      // Check that contents of arrays is identical
      if (_asArr1[i] && set1 && !set1->equals(*_asArr1[i])) match=kFALSE ;
      if (_asArr2[i] && set2 && !set2->equals(*_asArr2[i])) match=kFALSE ;	
      if (_asArr3[i] && set3 && !set3->equals(*_asArr3[i])) match=kFALSE ;	
      if (_asArr4[i] && set4 && !set4->equals(*_asArr4[i])) match=kFALSE ;	

      if (match) {
	if (set1) delete set1 ;
	if (set2) delete set2 ;
	if (set3) delete set3 ;
	if (set4) delete set4 ;
	return i ;
      }
    }
  }

  oocoutF((TObject*)0,Caching) << "RooAICRegistry::store: ERROR: capacity exceeded" << endl ;
  assert(0) ;
  return 0 ;
}



//_____________________________________________________________________________
const Int_t* RooAICRegistry::retrieve(Int_t masterCode) const 
{
  // Retrieve the array of integer codes associated with the given master code
  return _clArr[masterCode] ;
}



//_____________________________________________________________________________
const Int_t* RooAICRegistry::retrieve(Int_t masterCode, pRooArgSet& set1) const 
{
  // Retrieve the array of integer codes associated with the given master code
  // and set the passed set1 pointer to the first RooArgSet associated with this master code

  set1 = _asArr1[masterCode] ;
  return _clArr[masterCode] ;
}



//_____________________________________________________________________________
const Int_t* RooAICRegistry::retrieve(Int_t masterCode, pRooArgSet& set1, pRooArgSet& set2) const 
{
  // Retrieve the array of integer codes associated with the given master code
  // and set the passed set1,set2 pointers to the first and second  RooArgSets associated with this 
  // master code respectively

  set1 = _asArr1[masterCode] ;
  set2 = _asArr2[masterCode] ;
  return _clArr[masterCode] ;
}



//_____________________________________________________________________________
const Int_t* RooAICRegistry::retrieve(Int_t masterCode, pRooArgSet& set1, pRooArgSet& set2, pRooArgSet& set3, pRooArgSet& set4) const 
{
  // Retrieve the array of integer codes associated with the given master code
  // and set the passed set1-4 pointers to the four  RooArgSets associated with this 
  // master code respectively

  set1 = _asArr1[masterCode] ;
  set2 = _asArr2[masterCode] ;
  set3 = _asArr3[masterCode] ;
  set4 = _asArr4[masterCode] ;
  return _clArr[masterCode] ;
}
