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
\file RooObjCacheManager.cxx
\class RooObjCacheManager
\ingroup Roofitcore

Class RooObjCacheManager is an implementation of class RooCacheManager<RooAbsCacheElement>
and specializes in the storage of cache elements that contain RooAbsArg objects.
Caches with RooAbsArg derived payload require special care as server redirects
cache operation mode changes and constant term optimization calls may need to be
forwarded to such cache payload. This cache manager takes care of all these operations
by forwarding these calls to the RooAbsCacheElement interface functions, which
have a sensible default implementation. 
**/

#include "RooFit.h"
#include "Riostream.h"
#include <vector>
#include "RooObjCacheManager.h"
#include "RooMsgService.h"

using namespace std ;

ClassImp(RooObjCacheManager);
   ;


Bool_t RooObjCacheManager::_clearObsList(kFALSE) ;

////////////////////////////////////////////////////////////////////////////////
/// Constructor of object cache manager for given owner. If clearCacheOnServerRedirect is true
/// all cache elements will be cleared when a server redirect is intercepted by the cache manager.
/// This is the default strategy and should only be overridden when you really understand
/// what you're doing as properly implementing server redirect in cache elements can get very
/// complicated, especially if there are (cyclical) reference back to the owning object

RooObjCacheManager::RooObjCacheManager(RooAbsArg* owner, Int_t maxSize, Bool_t clearCacheOnServerRedirect, Bool_t allowOptimize) : 
  RooCacheManager<RooAbsCacheElement>(owner,maxSize), 
  _clearOnRedirect(clearCacheOnServerRedirect), 
  _allowOptimize(allowOptimize),
  _optCacheModeSeen(kFALSE),
  _optCacheObservables(0)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooObjCacheManager::RooObjCacheManager(const RooObjCacheManager& other, RooAbsArg* owner) : 
  RooCacheManager<RooAbsCacheElement>(other,owner),
  _clearOnRedirect(other._clearOnRedirect),
  _allowOptimize(other._allowOptimize),
  _optCacheModeSeen(kFALSE), // cache mode properties are not transferred in copy ctor
  _optCacheObservables(0)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooObjCacheManager::~RooObjCacheManager()
{
  if (_optCacheObservables) {
    delete _optCacheObservables ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Intercept server redirect calls. If clearOnRedirect was set, sterilize
/// the cache (i.e. keep the structure but delete all contents). If not
/// forward serverRedirect to cache elements

Bool_t RooObjCacheManager::redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) 
{ 
  if (_clearOnRedirect) {

    sterilize() ;
    
  } else {

    for (Int_t i=0 ; i<cacheSize() ; i++) {
      if (_object[i]) {
	_object[i]->redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
      }
    }

  }

  return kFALSE ; 
} 



////////////////////////////////////////////////////////////////////////////////
/// Intercept changes to cache operation mode and forward to cache elements

void RooObjCacheManager::operModeHook() 
{
  if (!_owner) {
    return ;
  }

  for (Int_t i=0 ; i<cacheSize() ; i++) {
    if (_object[i]) {
      _object[i]->operModeHook(_owner->operMode()) ;
    }
  }
} 



////////////////////////////////////////////////////////////////////////////////
/// Intercept calls to perform automatic optimization of cache mode operation. 
/// Forward calls to existing cache elements and save configuration of
/// cache mode optimization so that it can be applied on new cache elements
/// upon insertion 

void RooObjCacheManager::optimizeCacheMode(const RooArgSet& obs, RooArgSet& optNodes, RooLinkedList& processedNodes) 
{
  oocxcoutD(_owner,Caching) << "RooObjCacheManager::optimizeCacheMode(owner=" << _owner->GetName() << ") obs = " << obs << endl ;

  _optCacheModeSeen = kTRUE ;

  if (_optCacheObservables) {
    _optCacheObservables->removeAll() ;
    _optCacheObservables->add(obs) ;
  } else {
    _optCacheObservables = (RooArgSet*) new RooArgSet(obs) ;
  }
  
  for (Int_t i=0 ; i<cacheSize() ; i++) {
    if (_object[i]) {
      _object[i]->optimizeCacheMode(obs,optNodes,processedNodes) ;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooObjCacheManager::sterilize() 
{
  RooCacheManager<RooAbsCacheElement>::sterilize() ;

  // WVE - adding this causes trouble with IntegralMorpht????
  // Perhaps this should not be done in sterilize, but be a separate operation
  // called specially from RooAbsObsTestStatistic::setData()

  if (_optCacheObservables && _clearObsList) {
    delete _optCacheObservables ;
    _optCacheObservables = 0 ;
    _optCacheModeSeen = kFALSE ;
  }
  
}



////////////////////////////////////////////////////////////////////////////////
/// Set owner link on all object inserted into cache.
/// Also if cache mode optimization was requested, apply
/// it now to cache element being inserted

void RooObjCacheManager::insertObjectHook(RooAbsCacheElement& obj) 
{
  obj.setOwner(_owner) ;

  // If value caching mode optimization has happened, process it now on object being inserted
  if (_optCacheModeSeen) {
    RooLinkedList l ;
    RooArgSet s ;
    obj.optimizeCacheMode(*_optCacheObservables,s,l) ;
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Add details on cache contents when printing in tree mode

void RooObjCacheManager::printCompactTreeHook(std::ostream& os, const char *indent)
{
  for (Int_t i=0 ; i<cacheSize() ; i++) {
    if (_object[i]) {
      _object[i]->printCompactTreeHook(os,indent,i,cacheSize()-1) ;
    }
  }  
}



////////////////////////////////////////////////////////////////////////////////
/// If clearOnRedirect is false, forward constant term optimization calls to
/// cache elements

void RooObjCacheManager::findConstantNodes(const RooArgSet& obs, RooArgSet& cacheList, RooLinkedList& processedNodes) 
{
  if (!_allowOptimize) {
    return ;
  }
  
  for (Int_t i=0 ; i<cacheSize() ; i++) {
    if (_object[i]) {
      _object[i]->findConstantNodes(obs,cacheList, processedNodes) ;
    }
  }
}


