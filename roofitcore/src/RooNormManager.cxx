/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooNormManager.cxx,v 1.13 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// Class RooNormManager manages a cache for normalization/projection integrals
// for RooAbsPdf objects. Normalization/projection integrals are always defined by to
// RooArgSet pointer containing the set of normalization / projection observables respectively.
//
// For efficiency reasons these pointer are derefenced as little as possible. This
// class contains a lookup table for RooArgSet pointer pairs -> normalization integrals
// Distinct pointer pairs that represent the same normalization/projection are recognized
// and will all point to the same normalization object. Up to 'maxSize' different normalization/
// projection configurations can be cached.
// 

#include "RooFit.h"

#include "RooNormManager.h"
#include "RooNormManager.h"

ClassImp(RooNormManager)
  ;


RooNormManager::RooNormManager(Int_t maxSize) 
{
  _maxSize = maxSize ;
  _size = 0 ;

  _nsetCache = new RooNormSetCache[maxSize] ;
  _norm = new pRooAbsReal[maxSize] ;

  _lastNorm = 0 ;
  _lastNormSet = 0 ;
  _lastNameSet = 0 ;

  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    _norm[i]=0 ;
  }
}


RooNormManager::RooNormManager(const RooNormManager& other) 
{
  _maxSize = other._maxSize ;
  _size = 0 ;
  
  _nsetCache = new RooNormSetCache[_maxSize] ;
  _norm = new pRooAbsReal[_maxSize] ;

  _lastNorm = 0 ;
  _lastNormSet = 0 ;
  _lastNameSet = 0 ;

  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    _norm[i]=0 ;
  }
}


RooNormManager::~RooNormManager()
{
  delete[] _nsetCache ;  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    delete _norm[i] ;
  }
  delete[] _norm ;
}
  

RooAbsArg* RooNormManager::getNormByIndex(Int_t index) const 
{
  if (index<0||index>=_size) {
    cout << "RooNormManager::getNormByIndex: ERROR index (" 
	 << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return _norm[index] ;
}


void RooNormManager::setNormalization(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, const TNamed* isetRangeName, RooAbsReal* norm) 
{
  // Check if normalization is already registered
  if (getNormalization(self,nset,iset,isetRangeName)) {
    cout << "RooNormManager::setNormalization(" << self->GetName() << "): normalization already registered" << endl ;
    return ;
  }

  if (_size==_maxSize) {
    cout << "RooNormManager::setNormalization(" << self->GetName() << "): cache is full" << endl ;
    return ;
  }

  _nsetCache[_size].autoCache(self,nset,iset,isetRangeName,kTRUE) ;
  if (_norm[_size]) {
    cout << "RooNormManager::setNormalization(" << self->GetName() << "): deleting previous normalization of slot " << _size << endl ;
    delete _norm[_size] ;
  }
//    cout << "RooNormManager::setNormalization(" << self->GetName() << "): storing normalization in slot " 
//         << _size << ":" << norm << "=" << norm->GetName() << endl ;
  _norm[_size] = norm ;
  _size++ ;

  if (_lastNorm==0) {
    _lastNorm = norm ;
    _lastNormSet = (RooArgSet*) nset ;
    _lastNameSet = (RooNameSet*) &_nsetCache[_size].nameSet1() ;    
  }
}


RooAbsReal* RooNormManager::getNormalization(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, const TNamed* isetRangeName) 
{
  Int_t i ;

  // Pass 1 -- Just see if this iset/nset is known
  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].contains(nset,iset,isetRangeName)==kTRUE) {
      _lastNorm = _norm[i] ;
      _lastNormSet = (RooArgSet*) nset ;
      _lastNameSet = (RooNameSet*) &_nsetCache[i].nameSet1() ;
      return _norm[i] ;
    }
  }

  // Pass 2 --- Check if it is compatible with any existing normalization
  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].autoCache(self,nset,iset,isetRangeName,kFALSE)==kFALSE) {
      _lastNorm = _norm[i] ;
      _lastNormSet = (RooArgSet*) nset ;
      _lastNameSet = (RooNameSet*) &_nsetCache[i].nameSet1() ;
      return _norm[i] ;
    }
  }
  return 0 ;
}


const RooNameSet* RooNormManager::findNameSet(const RooArgSet* nset) const 
{
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].containsSet1(nset)) {
      return &_nsetCache[i].nameSet1() ;
    }
  }
  return 0 ;
}

