/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_CACHE_MANAGER
#define ROO_CACHE_MANAGER

#include "Rtypes.h"

#include "Riostream.h"
#include "RooMsgService.h"
#include "RooNormSetCache.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooAbsCache.h"
#include "RooAbsCacheElement.h"
#include <vector>

class RooNameSet ;


template<class T>
class RooCacheManager : public RooAbsCache {

public:

  RooCacheManager(Int_t maxSize=10) ;
  RooCacheManager(RooAbsArg* owner, Int_t maxSize=10) ;
  RooCacheManager(const RooCacheManager& other, RooAbsArg* owner=0) ;
  virtual ~RooCacheManager() ;
  
  T* getObj(const RooArgSet* nset, Int_t* sterileIndex=0, const TNamed* isetRangeName=0) {
    return getObj(nset,0,sterileIndex,isetRangeName) ;
  }
  Int_t setObj(const RooArgSet* nset, T* obj, const TNamed* isetRangeName=0) {
    return setObj(nset,0,obj,isetRangeName) ;
  }

  T* getObj(const RooArgSet* nset, const RooArgSet* iset, Int_t* sterileIndex=0, const TNamed* isetRangeName=0) ;
  Int_t setObj(const RooArgSet* nset, const RooArgSet* iset, T* obj, const TNamed* isetRangeName=0) ;  

  void reset() ;
  void sterilize() ;

  Int_t lastIndex() const { return _lastIndex ; }
  Int_t cacheSize() const { return _size ; }

  virtual Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) { return kFALSE ; }
  virtual void operModeHook() {}
  virtual void printCompactTreeHook(std::ostream&, const char *) {} 

  T* getObjByIndex(Int_t index) const ;
  const RooNameSet* nameSet1ByIndex(Int_t index) const ;
  const RooNameSet* nameSet2ByIndex(Int_t index) const ;

  virtual void insertObjectHook(T&) {} 
 
protected:

  Int_t _maxSize ;
  Int_t _size ;
  Int_t _lastIndex ;

  RooNormSetCache* _nsetCache ; //!
  T** _object ; //!

  ClassDef(RooCacheManager,1) // Cache Manager class generic objects
} ;


// needed to avoid 'specialization after instantiation' problem in gcc 
template <> void RooCacheManager<RooAbsCacheElement>::ShowMembers(TMemberInspector &R__insp, char *R__parent) ;
template <> void RooCacheManager<std::vector<Double_t> >::ShowMembers(TMemberInspector &R__insp, char *R__parent) ;


template<class T>
RooCacheManager<T>::RooCacheManager(Int_t maxSize) : RooAbsCache(0)
{
  _maxSize = maxSize ;
  _nsetCache = new RooNormSetCache[maxSize] ;
  _object = new T*[maxSize] ;
}

template<class T>
RooCacheManager<T>::RooCacheManager(RooAbsArg* owner, Int_t maxSize) : RooAbsCache(owner)
{

  _maxSize = maxSize ;
  _size = 0 ;

  _nsetCache = new RooNormSetCache[maxSize] ;
  _object = new T*[maxSize] ;
  _lastIndex = -1 ;

  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    _object[i]=0 ;
  }

}


template<class T>
RooCacheManager<T>::RooCacheManager(const RooCacheManager& other, RooAbsArg* owner) : RooAbsCache(other,owner)
{
  _maxSize = other._maxSize ;
  _size = other._size ;
  
  _nsetCache = new RooNormSetCache[_maxSize] ;
  _object = new T*[_maxSize] ;
  _lastIndex = -1 ;

//   cout << "RooCacheManager:cctor(" << this << ")" << endl ;

  Int_t i ;
  for (i=0 ; i<other._size ; i++) {    
    _nsetCache[i].initialize(other._nsetCache[i]) ;
    _object[i] = 0 ;
  }

  for (i=other._size ; i<_maxSize ; i++) {    
    _object[i] = 0 ;
  }
}


template<class T>
RooCacheManager<T>::~RooCacheManager()
{

  delete[] _nsetCache ;  
  Int_t i ;
  for (i=0 ; i<_size ; i++) {
    delete _object[i] ;
  }
  delete[] _object ;
}


template<class T>
void RooCacheManager<T>::reset() 
{
  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    delete _object[i] ;
    _object[i]=0 ;
    _nsetCache[i].clear() ;
  }  
  _lastIndex = -1 ;
  _size = 0 ;
}
  


template<class T>
void RooCacheManager<T>::sterilize() 
{
  Int_t i ;
  for (i=0 ; i<_maxSize ; i++) {
    delete _object[i] ;
    _object[i]=0 ;
  }  
}
  


template<class T>
Int_t RooCacheManager<T>::setObj(const RooArgSet* nset, const RooArgSet* iset, T* obj, const TNamed* isetRangeName) 
{
  // Check if object is already registered
  Int_t sterileIdx(-1) ;
  if (getObj(nset,iset,&sterileIdx,isetRangeName)) {
    //cout << "RooCacheManager::setNormList(" << self->GetName() << "): normalization list already registered" << endl ;
    return lastIndex() ;
  } 


  if (sterileIdx>=0) {
    // Found sterile slot that can should be recycled [ sterileIndex only set if isetRangeName matches ]
    //cout << "RooCacheManager::setNormList(" << self->GetName() << "): recycling sterile slot " << lastIndex() << endl ;
    _object[sterileIdx] = obj ;
    return lastIndex() ;
  }

  if (_size==_maxSize) {
    //cout << "RooCacheManager::setNormList(" << self->GetName() << "): cache is full" << endl ;
    return -1 ;
  }

  _nsetCache[_size].autoCache(_owner,nset,iset,isetRangeName,kTRUE) ;
  if (_object[_size]) {
    //cout << "RooCacheManager::setNormList(" << self->GetName() << "): deleting previous normalization list of slot " << _size << endl ;
    delete _object[_size] ;
  }

  _object[_size] = obj ;
  _size++ ;

  // Allow optional post-processing of object inserted in cache
  insertObjectHook(*obj) ;

  return _size-1 ;
}


template<class T>
T* RooCacheManager<T>::getObj(const RooArgSet* nset, const RooArgSet* iset, Int_t* sterileIdx, const TNamed* isetRangeName) 
{
  Int_t i ;

  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].contains(nset,iset,isetRangeName)==kTRUE) {      
      _lastIndex = i ;
      if(_object[i]==0 && sterileIdx) *sterileIdx=i ;
      return _object[i] ;
    }
  }

  for (i=0 ; i<_size ; i++) {
    if (_nsetCache[i].autoCache(_owner,nset,iset,isetRangeName,kFALSE)==kFALSE) {
      _lastIndex = i ;
      if(_object[i]==0 && sterileIdx) *sterileIdx=i ;
      return _object[i] ;
    }
  }
  return 0 ;
}



template<class T>
T* RooCacheManager<T>::getObjByIndex(Int_t index) const 
{
  if (index<0||index>=_size) {
    oocoutE(_owner,ObjectHandling) << "RooCacheManager::getNormListByIndex: ERROR index (" 
				   << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return _object[index] ;
}

template<class T>
const RooNameSet* RooCacheManager<T>::nameSet1ByIndex(Int_t index) const
{
  if (index<0||index>=_size) {
    oocoutE(_owner,ObjectHandling) << "RooCacheManager::getNormListByIndex: ERROR index (" 
				   << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return &_nsetCache[index].nameSet1() ;
}

template<class T>
const RooNameSet* RooCacheManager<T>::nameSet2ByIndex(Int_t index) const 
{
  if (index<0||index>=_size) {
    oocoutE(_owner,ObjectHandling) << "RooCacheManager::getNormListByIndex: ERROR index (" 
				   << index << ") out of range [0," << _size-1 << "]" << endl ;
    return 0 ;
  }
  return &_nsetCache[index].nameSet2() ;
}


#endif 
