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

// -- CLASS DESCRIPTION [AUX] --
// Class RooCacheManager manages a cache for lists of normalization/projection integrals
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
#include <vector>
#include "RooObjCacheManager.h"
#include "RooMsgService.h"

ClassImp(RooObjCacheManager)
   ;

RooObjCacheManager::RooObjCacheManager(RooAbsArg* owner, Int_t maxSize, Bool_t clearCacheOnServerRedirect) : 
  RooCacheManager<RooAbsCacheElement>(owner,maxSize), 
  _clearOnRedirect(clearCacheOnServerRedirect), 
  _optCacheModeSeen(kFALSE),
  _optCacheObservables(0)
{
}

RooObjCacheManager::RooObjCacheManager(const RooObjCacheManager& other, RooAbsArg* owner) : 
  RooCacheManager<RooAbsCacheElement>(other,owner),
  _clearOnRedirect(other._clearOnRedirect),
  _optCacheModeSeen(kFALSE), // cache mode properties are not transferred in copy ctor
  _optCacheObservables(0)
{
}

RooObjCacheManager::~RooObjCacheManager()
{
  if (_optCacheObservables) {
    delete _optCacheObservables ;
    _optCacheObservables=0 ;
  }
}

Bool_t RooObjCacheManager::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{ 
  
  if (_clearOnRedirect) {

    sterilize() ;
    
  } else {

    for (Int_t i=0 ; i<_size ; i++) {
      _object[i]->redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
    }

  }

  return kFALSE ; 
} 


void RooObjCacheManager::operModeHook() 
{
  if (!_owner) {
    return ;
  }

  for (Int_t i=0 ; i<_size ; i++) {
    if (_object[i]) {
      _object[i]->operModeHook(_owner->operMode()) ;
    }
  }
} 


void RooObjCacheManager::optimizeCacheMode(const RooArgSet& obs, RooArgSet& optNodes, RooLinkedList& processedNodes) 
{
  oocxcoutD(_owner,"Caching") << "RooObjCacheManager::optimizeCacheMode(owner=" << _owner->GetName() << ") obs = " << obs << endl ;

  _optCacheModeSeen = kTRUE ;
  RooArgSet* oldObs = 0 ;
  if (_optCacheObservables) {    
    oldObs = _optCacheObservables ; 
  }
  _optCacheObservables = (RooArgSet*) obs.snapshot() ;
  
  if (oldObs) {
    //cout << "deleting old _optCacheObservables" << endl ;
    delete oldObs ;
  }

  for (Int_t i=0 ; i<_size ; i++) {
    if (_object[i]) {
      _object[i]->optimizeCacheMode(obs,optNodes,processedNodes) ;
    }
  }
}



void RooObjCacheManager::insertObjectHook(RooAbsCacheElement& obj) 
{
  //cout << "RooObjCacheManager::insertObjectHook(owner = " << _owner->GetName() << ")" << endl ;

  // If value caching mode optimization has happened, process it now on object being inserted
  if (_optCacheModeSeen) {
    //cout << "performing posterior optimization with _optCacheObservables = " << *_optCacheObservables << endl ;
    RooLinkedList l ;
    RooArgSet s ;
    obj.optimizeCacheMode(*_optCacheObservables,s,l) ;
  }

  // If oper mode is not default (Auto), then pass it on to element being inserted
  //   if (_owner->operMode()!=RooAbsArg::Auto) {
  //     obj.operModeHook(_owner->operMode()) ;
  //   }
}




void RooObjCacheManager::printCompactTreeHook(std::ostream& os, const char *indent)
{
  for (Int_t i=0 ; i<_size ; i++) {
    if (_object[i]) {
      _object[i]->printCompactTreeHook(os,indent,i,_size-1) ;
    }
  }  
}


void RooObjCacheManager::findConstantNodes(const RooArgSet& obs, RooArgSet& cacheList, RooLinkedList& processedNodes) 
{
  for (Int_t i=0 ; i<_size ; i++) {
    if (_object[i]) {
      _object[i]->findConstantNodes(obs,cacheList, processedNodes) ;
    }
  }    
}


