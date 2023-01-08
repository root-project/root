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
\file RooExpensiveObjectCache.cxx
\class RooExpensiveObjectCache
\ingroup Roofitcore

RooExpensiveObjectCache is a singleton class that serves as repository
for objects that are expensive to calculate. Owners of such objects
can registers these here with associated parameter values for which
the object is valid, so that other instances can, at a later moment
retrieve these precalculated objects.
**/

#include "RooExpensiveObjectCache.h"

#include "TClass.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include <iostream>

ClassImp(RooExpensiveObjectCache);
ClassImp(RooExpensiveObjectCache::ExpensiveObject);


////////////////////////////////////////////////////////////////////////////////
/// Destructor.

RooExpensiveObjectCache::~RooExpensiveObjectCache()
{
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter!=_map.end() ; ++iter) {
    delete iter->second ;
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Return reference to singleton instance

RooExpensiveObjectCache& RooExpensiveObjectCache::instance()
{
  static RooExpensiveObjectCache instance;
  return instance;
}


////////////////////////////////////////////////////////////////////////////////
/// Register object associated with given name and given associated parameters with given values in cache.
/// The cache will take _ownership_of_object_ and is indexed under the given name (which does not
/// need to be the name of cacheObject and with given set of dependent parameters with validity for the
/// current values of those parameters. It can be retrieved later by callin retrieveObject()

bool RooExpensiveObjectCache::registerObject(const char* ownerName, const char* objectName, TObject& cacheObject, const RooArgSet& params)
{
  // Delete any previous object
  ExpensiveObject* eo = _map[objectName] ;
  Int_t olduid(-1) ;
  if (eo) {
    olduid = eo->uid() ;
    delete eo ;
  }
  // Install new object
  _map[objectName] = new ExpensiveObject(olduid!=-1?olduid:_nextUID++, ownerName,cacheObject,params) ;

  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Retrieve object from cache that was registered under given name with given parameters, _if_
/// current parameter values match those that were stored in the registry for this object.
/// The return object is owned by the cache instance.

const TObject* RooExpensiveObjectCache::retrieveObject(const char* name, TClass* tc, const RooArgSet& params)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Retrieve payload object of cache element with given unique ID

const TObject* RooExpensiveObjectCache::getObj(Int_t uid)
{
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; ++iter) {
    if (iter->second->uid() == uid) {
      return iter->second->payload() ;
    }
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Clear cache element with given unique ID
/// Retrieve payload object of cache element with given unique ID

bool RooExpensiveObjectCache::clearObj(Int_t uid)
{
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; ++iter) {
    if (iter->second->uid() == uid) {
      _map.erase(iter->first) ;
      return false ;
    }
  }
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Place new payload object in cache element with given unique ID. Cache
/// will take ownership of provided object!

bool RooExpensiveObjectCache::setObj(Int_t uid, TObject* obj)
{
  for (std::map<TString,ExpensiveObject*>::iterator iter = _map.begin() ; iter !=_map.end() ; ++iter) {
    if (iter->second->uid() == uid) {
      iter->second->setPayload(obj) ;
      return false ;
    }
  }
  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Clear all cache elements

void RooExpensiveObjectCache::clearAll()
{
  _map.clear() ;
}






////////////////////////////////////////////////////////////////////////////////
/// Construct ExpensiveObject oject for inPayLoad and store reference values
/// for all RooAbsReal and RooAbsCategory parameters in params.

RooExpensiveObjectCache::ExpensiveObject::ExpensiveObject(Int_t uidIn, const char* inOwnerName, TObject& inPayload, RooArgSet const& params)
{
  _uid = uidIn ;
  _ownerName = inOwnerName;

  _payload = &inPayload ;

  for(RooAbsArg * arg : params) {
    RooAbsReal* real = dynamic_cast<RooAbsReal*>(arg) ;
    if (real) {
      _realRefParams[real->GetName()] = real->getVal() ;
    } else {
      RooAbsCategory* cat = dynamic_cast<RooAbsCategory*>(arg) ;
      if (cat) {
   _catRefParams[cat->GetName()] = cat->getCurrentIndex() ;
      } else {
   oocoutW(&inPayload,Caching) << "RooExpensiveObject::registerObject() WARNING: ignoring non-RooAbsReal/non-RooAbsCategory reference parameter " << arg->GetName() << std::endl ;
      }
    }
  }

}



////////////////////////////////////////////////////////////////////////////////

RooExpensiveObjectCache::ExpensiveObject::ExpensiveObject(Int_t uidIn, const ExpensiveObject& other) :
  _uid(uidIn),
  _realRefParams(other._realRefParams),
  _catRefParams(other._catRefParams),
  _ownerName(other._ownerName)
{
  _payload = other._payload->Clone() ;
}



////////////////////////////////////////////////////////////////////////////////

RooExpensiveObjectCache::ExpensiveObject::~ExpensiveObject()
{
  delete _payload ;
}





////////////////////////////////////////////////////////////////////////////////
/// Check object type ;

bool RooExpensiveObjectCache::ExpensiveObject::matches(TClass* tc, const RooArgSet& params)
{
  if (_payload->IsA() != tc) {
    return false;
  }

  // Check parameters
  for(RooAbsArg * arg : params) {
    RooAbsReal* real = dynamic_cast<RooAbsReal*>(arg) ;
    if (real) {
      if (std::abs(real->getVal()-_realRefParams[real->GetName()])>1e-12) {
   return false ;
      }
    } else {
      RooAbsCategory* cat = dynamic_cast<RooAbsCategory*>(arg) ;
      if (cat) {
   if (cat->getCurrentIndex() != _catRefParams[cat->GetName()]) {
     return false ;
   }
      }
    }
  }

  return true ;

}



////////////////////////////////////////////////////////////////////////////////

void RooExpensiveObjectCache::print() const
{
  for(auto const& item : _map) {
    std::cout << "uid = " << item.second->uid() << " key=" << item.first << " value=" ;
    item.second->print() ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooExpensiveObjectCache::ExpensiveObject::print() const
{
  std::cout << _payload->ClassName() << "::" << _payload->GetName() ;
  if (_realRefParams.size()>0 || _catRefParams.size()>0) {
    std::cout << " parameters=( " ;
    auto iter = _realRefParams.begin() ;
    while(iter!=_realRefParams.end()) {
      std::cout << iter->first << "=" << iter->second << " " ;
      ++iter ;
    }
    auto iter2 = _catRefParams.begin() ;
    while(iter2!=_catRefParams.end()) {
      std::cout << iter2->first << "=" << iter2->second << " " ;
      ++iter2 ;
    }
    std::cout << ")" ;
  }
  std::cout << std::endl ;
}




////////////////////////////////////////////////////////////////////////////////

void RooExpensiveObjectCache::importCacheObjects(RooExpensiveObjectCache& other, const char* ownerName, bool verbose)
{
  for(auto const& item : other._map) {
    if (std::string(ownerName)==item.second->ownerName()) {
      _map[item.first.Data()] = new ExpensiveObject(_nextUID++, *item.second) ;
      if (verbose) {
   oocoutI(item.second->payload(),Caching) << "RooExpensiveObjectCache::importCache() importing cache object "
                   << item.first << " associated with object " << item.second->ownerName() << std::endl ;
      }
    }
  }

}
