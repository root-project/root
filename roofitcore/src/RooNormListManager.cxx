/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooNormListManager.cxx,v 1.15 2007/05/11 09:11:58 verkerke Exp $
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
// Class RooNormListManager manages a cache for lists of normalization/projection integrals
// for RooAbsPdf objects. Normalization/projection integrals are always defined by to
// RooArgSet pointer containing the set of normalization / projection observables respectively.
//
// For efficiency reasons these pointer are derefenced as little as possible. This
// class contains a lookup table for RooArgSet pointer pairs -> normalization lists.
// Distinct pointer pairs that represent the same normalization/projection are recognized
// and will all point to the same normalization list. Lists for up to 'maxSize' different normalization/
// projection configurations can be cached.
// 

#include "RooFit.h"

#include "RooNormListManager.h"
#include "RooNormListManager.h"

ClassImp(RooNormListManager)
  ;


Bool_t RooNormListManager::_verbose(kFALSE) ;


RooNormListManager::RooNormListManager(Int_t maxSize) 
{
  _maxSize = maxSize ;
  _size = 0 ;

  _nsetCache = new RooNormSetCache[maxSize] ;
  _normList = new pRooArgList[maxSize] ;
  _lastIndex = -1 ;

  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    _normList[i]=0 ;
  }
}


RooNormListManager::RooNormListManager(const RooNormListManager& other, Bool_t sterileCopy) 
{
  _maxSize = other._maxSize ;
  _size = other._size ;
  
  _nsetCache = new RooNormSetCache[_maxSize] ;
  _normList = new pRooArgList[_maxSize] ;
  _lastIndex = -1 ;

//   cout << "RooNormListManager:cctor(" << this << ")" << endl ;

  Int_t i ;
  for (i=0 ; i<other._size ; i++) {    
    _nsetCache[i].initialize(other._nsetCache[i]) ;

    if (!sterileCopy) {
      if (other._normList[i]->isOwning()) {
	_normList[i] = new RooArgList ;
	TIterator* iter = other._normList[i]->createIterator() ;
	RooAbsArg* arg ;
	while((arg=(RooAbsArg*)iter->Next())) {
	  RooAbsArg* argclone = (RooAbsArg*)arg->Clone() ;
	  _normList[i]->addOwned(*argclone) ;
	}
	delete iter ;
	
      } else {
	_normList[i] = (RooArgList*) other._normList[i]->Clone() ;
      }
    } else {
      _normList[i] = 0 ;
    }
  }

  for (i=other._size ; i<_maxSize ; i++) {    
    _normList[i] = 0 ;
  }
}


RooNormListManager::~RooNormListManager()
{
  delete[] _nsetCache ;  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    delete _normList[i] ;
  }
  delete[] _normList ;
}


void RooNormListManager::reset() 
{
  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    delete _normList[i] ;
    _normList[i]=0 ;
    _nsetCache[i].clear() ;
  }  
  _lastIndex = -1 ;
  _size = 0 ;
}
  


void RooNormListManager::sterilize() 
{
  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    delete _normList[i] ;
    _normList[i]=0 ;
  }  
}
  


Int_t RooNormListManager::setNormList(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, RooArgList* normList, const TNamed* isetRangeName) 
{
  // Check if normalization is already registered
  Int_t sterileIdx(-1) ;
  if (getNormList(self,nset,iset,&sterileIdx,isetRangeName)) {
    //cout << "RooNormListManager::setNormList(" << self->GetName() << "): normalization list already registered" << endl ;
    return lastIndex() ;
  } 

  if (sterileIdx>=0) {
    // Found sterile slot that can should be recycled [ sterileIndex only set if isetRangeName matches ]
    //cout << "RooNormListManager::setNormList(" << self->GetName() << "): recycling sterile slot " << lastIndex() << endl ;
    _normList[sterileIdx] = normList ;
    return lastIndex() ;
  }

  if (_size==_maxSize) {
    //cout << "RooNormListManager::setNormList(" << self->GetName() << "): cache is full" << endl ;
    return -1 ;
  }

  _nsetCache[_size].autoCache(self,nset,iset,isetRangeName,kTRUE) ;
  if (_normList[_size]) {
    //cout << "RooNormListManager::setNormList(" << self->GetName() << "): deleting previous normalization list of slot " << _size << endl ;
    delete _normList[_size] ;
  }
  if (_verbose) {
    cout << "RooNormListManager::setNormList(" << self->GetName() << "): storing normalization list in slot " 
	 << _size << ":" << normList << "=" << normList->GetName() << " nset=" ;
    if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
  }

  _normList[_size] = normList ;
  _size++ ;

  return _size-1 ;
}


RooArgList* RooNormListManager::getNormList(const RooAbsArg* self, const RooArgSet* nset, const RooArgSet* iset, 
					    Int_t* sterileIdx, const TNamed* isetRangeName) 
{
  Int_t i ;

  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].contains(nset,iset,isetRangeName)==kTRUE) {      
      _lastIndex = i ;
      if(_normList[i]==0 && sterileIdx) *sterileIdx=i ;
      return _normList[i] ;
    }
  }

  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].autoCache(self,nset,iset,isetRangeName,kFALSE)==kFALSE) {
      _lastIndex = i ;
      if(_normList[i]==0 && sterileIdx) *sterileIdx=i ;
      return _normList[i] ;
    }
  }
  return 0 ;
}



RooArgList* RooNormListManager::getNormListByIndex(Int_t index) const 
{
  if (index<0||index>=_size) {
    cout << "RooNormListManager::getNormListByIndex: ERROR index (" 
	 << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return _normList[index] ;
}

const RooNameSet* RooNormListManager::nameSet1ByIndex(Int_t index) const
{
  if (index<0||index>=_size) {
    cout << "RooNormListManager::getNormListByIndex: ERROR index (" 
	 << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return &_nsetCache[index].nameSet1() ;
}

const RooNameSet* RooNormListManager::nameSet2ByIndex(Int_t index) const 
{
  if (index<0||index>=_size) {
    cout << "RooNormListManager::getNormListByIndex: ERROR index (" 
	 << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return &_nsetCache[index].nameSet2() ;
}
