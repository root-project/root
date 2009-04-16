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
// RooExpensiveObjectCache is a singleton class that serves as repository
// for objects that are expensive to calculate. Owners of such objects
// can registers these here with associated parameter values for which
// the object is valid, so that other instances can, at a later moment
// retrieve these precalculated objects

// END_HTML
//


#include "TClass.h"
#include "RooFit.h"
#include "RooSentinel.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include <iostream>
#include <math.h>
using namespace std ;

#include "RooExpensiveObjectCache.h"

ClassImp(RooExpensiveObjectCache)
ClassImp(RooExpensiveObjectCache::ExpensiveObject)
  ;

RooExpensiveObjectCache* RooExpensiveObjectCache::_instance = 0 ;


//_____________________________________________________________________________
RooExpensiveObjectCache::RooExpensiveObjectCache() : _nextUID(0)
{
  // Constructor
}



//_____________________________________________________________________________
RooExpensiveObjectCache::RooExpensiveObjectCache(const RooExpensiveObjectCache& other) :
  TObject(other), _nextUID(0)
{
  // Copy constructor
}



//_____________________________________________________________________________
RooExpensiveObjectCache::~RooExpensiveObjectCache() 
{
  // Destructor. 

  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter!=_map.end() ; ++iter) {
    delete iter->second ;
  }

  if (_instance == this) {
    _instance = 0 ;
  }
}

 


//_____________________________________________________________________________
RooExpensiveObjectCache& RooExpensiveObjectCache::instance() 
{
  // Return reference to singleton instance 

  if (!_instance) {
    _instance = new RooExpensiveObjectCache() ;    
    RooSentinel::activate() ;    
  }
  return *_instance ;
}




//_____________________________________________________________________________
void RooExpensiveObjectCache::cleanup() 
{
  // Static function called by RooSentinel atexit() handler to cleanup at end of program
  delete _instance ;
}




//_____________________________________________________________________________
Bool_t RooExpensiveObjectCache::registerObject(const char* ownerName, const char* objectName, TObject& cacheObject, const RooArgSet& params) 
{
  // Register object associated with given name and given associated parameters with given values in cache.
  // The cache will take _ownership_of_object_ and is indexed under the given name (which does not
  // need to be the name of cacheObject and with given set of dependent parameters with validity for the
  // current values of those parameters. It can be retrieved later by callin retrieveObject()

  TIterator* parIter = params.createIterator() ;
  Bool_t ret = registerObject(ownerName,objectName,cacheObject,parIter) ;
  delete parIter ;

  return ret ;
}



//_____________________________________________________________________________
Bool_t RooExpensiveObjectCache::registerObject(const char* ownerName, const char* objectName, TObject& cacheObject, TIterator* parIter) 
{
  // Register object associated with given name and given associated parameters with given values in cache.
  // The cache will take _ownership_of_object_ and is indexed under the given name (which does not
  // need to be the name of cacheObject and with given set of dependent parameters with validity for the
  // current values of those parameters. It can be retrieved later by callin retrieveObject()

  // Delete any previous object
  ExpensiveObject* eo = _map[objectName] ;
  Int_t olduid(-1) ;
  if (eo) {
    olduid = eo->uid() ;
    delete eo ;
  }
  // Install new object
  _map[objectName] = new ExpensiveObject(olduid!=-1?olduid:_nextUID++, ownerName,cacheObject,parIter) ;

  return kFALSE ;
}



//_____________________________________________________________________________
const TObject* RooExpensiveObjectCache::retrieveObject(const char* name, TClass* tc, const RooArgSet& params) 
{
  // Retrieve object from cache that was registered under given name with given parameters, _if_
  // current parameter values match those that were stored in the registry for this object.
  // The return object is owned by the cache instance.

  ExpensiveObject* eo = _map[name] ;

  // If no cache element found, return 0 ;
  if (!eo) {
    return 0 ;
  }
  
  // If parameters also match, return payload ;
  if (eo->matches(tc,params)) {
    return eo->payload() ;
  }

  return 0 ;
}



//_____________________________________________________________________________
const TObject* RooExpensiveObjectCache::getObj(Int_t uid) 
{
  // Retrieve payload object of cache element with given unique ID  
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; iter++) {
    if (iter->second->uid() == uid) {
      return iter->second->payload() ;
    }
  }  
  return 0 ;
}



//_____________________________________________________________________________
Bool_t RooExpensiveObjectCache::clearObj(Int_t uid) 
{
  // Clear cache element with given unique ID
  // Retrieve payload object of cache element with given unique ID  
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; iter++) {
    if (iter->second->uid() == uid) {
      _map.erase(iter->first) ;
      return kFALSE ;
    }
  }  
  return kTRUE ;  
}



//_____________________________________________________________________________
Bool_t RooExpensiveObjectCache::setObj(Int_t uid, TObject* obj) 
{
  // Place new payload object in cache element with given unique ID. Cache
  // will take ownership of provided object!

  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; iter++) {
    if (iter->second->uid() == uid) {
      iter->second->setPayload(obj) ;
      return kFALSE ;
    }
  }  
  return kTRUE ;
}



//_____________________________________________________________________________
void RooExpensiveObjectCache::clearAll() 
{
  // Clear all cache elements
  _map.clear() ;
}






//_____________________________________________________________________________
RooExpensiveObjectCache::ExpensiveObject::ExpensiveObject(Int_t uidIn, const char* inOwnerName, TObject& inPayload, TIterator* parIter) 
{
  // Construct ExpensiveObject oject for inPayLoad and store reference values
  // for all RooAbsReal and RooAbsCategory parameters in params.

  _uid = uidIn ;
  _ownerName = inOwnerName;

  _payload = &inPayload ;

  RooAbsArg* arg ;
  parIter->Reset() ;
  while((arg=(RooAbsArg*)parIter->Next() )) {
    RooAbsReal* real = dynamic_cast<RooAbsReal*>(arg) ;
    if (real) {
      _realRefParams[real->GetName()] = real->getVal() ;
    } else {
      RooAbsCategory* cat = dynamic_cast<RooAbsCategory*>(arg) ;
      if (cat) {
	_catRefParams[cat->GetName()] = cat->getIndex() ;
      } else {
	oocoutW(&inPayload,Caching) << "RooExpensiveObject::registerObject() WARNING: ignoring non-RooAbsReal/non-RooAbsCategory reference parameter " << arg->GetName() << endl ;
      }
    }
  }
  
}



//_____________________________________________________________________________
RooExpensiveObjectCache::ExpensiveObject::ExpensiveObject(Int_t uidIn, const ExpensiveObject& other) : 
  _uid(uidIn),
  _realRefParams(other._realRefParams), 
  _catRefParams(other._catRefParams),
  _ownerName(other._ownerName)
{
  _payload = other._payload->Clone() ;
}



//_____________________________________________________________________________
RooExpensiveObjectCache::ExpensiveObject::~ExpensiveObject() 
{
  delete _payload ;
}





//_____________________________________________________________________________
Bool_t RooExpensiveObjectCache::ExpensiveObject::matches(TClass* tc, const RooArgSet& params) 
{
  // Check object type ;
  if (_payload->IsA() != tc) {
    return kFALSE; 
  }

  // Check parameters 
  TIterator* iter = params.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next() )) {
    RooAbsReal* real = dynamic_cast<RooAbsReal*>(arg) ;
    if (real) {
      if (fabs(real->getVal()-_realRefParams[real->GetName()])>1e-12) {
	delete iter ;
	return kFALSE ;
      } 
    } else {
      RooAbsCategory* cat = dynamic_cast<RooAbsCategory*>(arg) ;
      if (cat) {
	if (cat->getIndex() != _catRefParams[cat->GetName()]) {
	  delete iter ;
	  return kFALSE ;
	}
      }
    }
  }
  delete iter ;

  return kTRUE ;

}



//_____________________________________________________________________________
void RooExpensiveObjectCache::print() const 
{
  map<TString,ExpensiveObject*>::const_iterator iter = _map.begin() ;

  while(iter!=_map.end()) {
    cout << "uid = " << iter->second->uid() << " key=" << iter->first << " value=" ;
    iter->second->print() ;    
    ++iter ;
  }
}



//_____________________________________________________________________________
void RooExpensiveObjectCache::ExpensiveObject::print() 
{
  cout << _payload->IsA()->GetName() << "::" << _payload->GetName() ;
  if (_realRefParams.size()>0 || _catRefParams.size()>0) {
    cout << " parameters=( " ;
    map<TString,Double_t>::iterator iter = _realRefParams.begin() ;
    while(iter!=_realRefParams.end()) {
      cout << iter->first << "=" << iter->second << " " ;
      ++iter ;
    }  
    map<TString,Int_t>::iterator iter2 = _catRefParams.begin() ;
    while(iter2!=_catRefParams.end()) {
      cout << iter2->first << "=" << iter2->second << " " ;
      ++iter2 ;
    }
    cout << ")" ;
  }
  cout << endl ;
}




//_____________________________________________________________________________
void RooExpensiveObjectCache::importCacheObjects(RooExpensiveObjectCache& other, const char* ownerName, Bool_t verbose) 
{
  map<TString,ExpensiveObject*>::const_iterator iter = other._map.begin() ;
  while(iter!=other._map.end()) {
    if (string(ownerName)==iter->second->ownerName()) {      
      _map[iter->first.Data()] = new ExpensiveObject(_nextUID++, *iter->second) ;
      if (verbose) {
	oocoutI(iter->second->payload(),Caching) << "RooExpensiveObjectCache::importCache() importing cache object " 
						 << iter->first << " associated with object " << iter->second->ownerName() << endl ;
      }
    }
    ++iter ;
  }
  
}
