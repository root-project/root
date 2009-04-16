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
// RooSetProxy is the concrete proxy for RooArgSet objects.
// A RooSetProxy is the general mechanism to store a RooArgSet
// with RooAbsArgs in a RooAbsArg.
//
// Creating a RooSetProxy adds all members of the proxied RooArgSet to the proxy owners
// server list (thus receiving value/shape dirty flags from it) and
// registers itself with the owning class. The latter allows the
// owning class to update the pointers of RooArgSet contents to reflect
// the serverRedirect changes.
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "RooSetProxy.h"
#include "RooSetProxy.h"
#include "RooArgSet.h"
#include "RooAbsArg.h"

ClassImp(RooSetProxy)
;


#ifdef USEMEMPOOL

//_____________________________________________________________________________
void* RooSetProxy::operator new (size_t bytes)
{
  // Overload new operator must be implemented because it is overloaded
  // in the RooArgSet base class. Perform standard memory allocation
  // here instead of memory pool management performed in RooArgSet

  return malloc(bytes) ;
}


//_____________________________________________________________________________
void RooSetProxy::operator delete (void* ptr)
{
  free(ptr) ;
}

#endif


//_____________________________________________________________________________
RooSetProxy::RooSetProxy(const char* inName, const char* /*desc*/, RooAbsArg* owner, 
			 Bool_t defValueServer, Bool_t defShapeServer) :
  RooArgSet(inName), _owner(owner), 
  _defValueServer(defValueServer), 
  _defShapeServer(defShapeServer)
{
  // Construct proxy with given name and description, with given owner
  // The default value and shape dirty propagation of the set contents
  // to the set owner is controlled by flags defValueServer and defShapeServer

  //SetTitle(desc) ;
  _owner->registerProxy(*this) ;
  _iter = createIterator() ;
}



//_____________________________________________________________________________
RooSetProxy::RooSetProxy(const char* inName, RooAbsArg* owner, const RooSetProxy& other) : 
  RooArgSet(other,inName), _owner(owner),  
  _defValueServer(other._defValueServer), 
  _defShapeServer(other._defShapeServer)
{
  // Copy constructor

  _owner->registerProxy(*this) ;
  _iter = createIterator() ;
}



//_____________________________________________________________________________
RooSetProxy::~RooSetProxy()
{
  // Destructor
  if (_owner) _owner->unRegisterProxy(*this) ;
  delete _iter ;
}



//_____________________________________________________________________________
Bool_t RooSetProxy::add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent)
{
  // Overloaded RooArgSet::add() method insert object into set 
  // and registers object as server to owner with given value
  // and shape dirty flag propagation requests

  Bool_t ret=RooArgSet::add(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,valueServer,shapeServer) ;
  }
  return ret ;  
}



//_____________________________________________________________________________
Bool_t RooSetProxy::addOwned(RooAbsArg& var, Bool_t silent)
{
  // Overloaded RooArgSet::addOwned() method insert object into owning set 
  // and registers object as server to owner with default value
  // and shape dirty flag propagation

  Bool_t ret=RooArgSet::addOwned(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,_defValueServer,_defShapeServer) ;
  }
  return ret ;  
}
				 


//_____________________________________________________________________________
RooAbsArg* RooSetProxy::addClone(const RooAbsArg& var, Bool_t silent) 
{
  // Overloaded RooArgSet::addClone() method insert clone of object into owning set 
  // and registers cloned object as server to owner with default value
  // and shape dirty flag propagation

  RooAbsArg* ret=RooArgSet::addClone(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,_defValueServer,_defShapeServer) ;
  }
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooSetProxy::add(const RooAbsArg& var, Bool_t silent) 
{
  // Overloaded RooArgSet::add() method inserts 'var' into set
  // and registers 'var' as server to owner with default value
  // and shape dirty flag propagation

  return add(var,_defValueServer,_defShapeServer,silent) ;
}



//_____________________________________________________________________________
Bool_t RooSetProxy::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  // Replace object 'var1' in set with 'var2'. Deregister var1 as
  // server from owner and register var2 as server to owner with
  // default value and shape dirty propagation flags

  Bool_t ret=RooArgSet::replace(var1,var2) ;
  if (ret) {
    if (!isOwning()) _owner->removeServer((RooAbsArg&)var1) ;
    _owner->addServer((RooAbsArg&)var2,_owner->isValueServer(var1),
		                       _owner->isShapeServer(var2)) ;
  }
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooSetProxy::remove(const RooAbsArg& var, Bool_t silent, Bool_t matchByNameOnly) 
{
  // Remove object 'var' from set and deregister 'var' as server to owner.

  Bool_t ret=RooArgSet::remove(var,silent,matchByNameOnly) ;
  if (ret && !isOwning()) {
    _owner->removeServer((RooAbsArg&)var) ;
  }
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooSetProxy::remove(const RooAbsCollection& list, Bool_t silent, Bool_t matchByNameOnly) 
{
  // Remove each argument in the input list from our list using remove(const RooAbsArg&).
  // and remove each argument as server to owner

  Bool_t result(false) ;

  TIterator* iter = list.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    result |= remove(*arg,silent,matchByNameOnly) ;
  }
  delete iter ;

  return result;  
}




//_____________________________________________________________________________
void RooSetProxy::removeAll() 
{
  // Remove all argument inset using remove(const RooAbsArg&).
  // and remove each argument as server to owner

  if (!isOwning()) {
    TIterator* iter = createIterator() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)iter->Next())) {
      if (!isOwning()) {
	_owner->removeServer(*arg) ;
      }
    }
    delete iter ;
  }

  RooArgSet::removeAll() ;
}




//_____________________________________________________________________________
RooSetProxy& RooSetProxy::operator=(const RooArgSet& other) 
{
  // Assign values of arguments on other set to arguments in this set
  RooArgSet::operator=(other) ;
  return *this ;
}




//_____________________________________________________________________________
Bool_t RooSetProxy::changePointer(const RooAbsCollection& newServerList, Bool_t nameChange, Bool_t factoryInitMode) 
{
  // Process server change operation on owner. Replace elements in set with equally
  // named objects in 'newServerList'

  if (getSize()==0) {
    if (factoryInitMode) {
      TIterator* iter = newServerList.createIterator() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)iter->Next())) {
	add(*arg,kTRUE) ;
      }
      delete iter ;
    } else {
      return kTRUE ;	
    }
  }

  _iter->Reset() ;
  RooAbsArg* arg ;
  Bool_t error(kFALSE) ;
  while ((arg=(RooAbsArg*)_iter->Next())) {
    
    RooAbsArg* newArg= arg->findNewServer(newServerList, nameChange);
    if (newArg) error |= !RooArgSet::replace(*arg,*newArg) ;
  }
  return !error ;
}



//_____________________________________________________________________________
void RooSetProxy::print(ostream& os, Bool_t addContents) const 
{ 
  // Printing name of proxy on ostream. If addContents is true
  // also print names of objects in set

  if (!addContents) {
    os << name() << "=" ; printStream(os,kValue,kInline) ; 
  } else {
    os << name() << "=(" ;
    TIterator* iter = createIterator() ;
    RooAbsArg* arg ;
    Bool_t first2(kTRUE) ;
    while ((arg=(RooAbsArg*)iter->Next())) {
      if (first2) {
	first2 = kFALSE ;
      } else {
	os << "," ;
      }
      arg->printStream(os,kValue|kName,kInline) ;
    }
    os << ")" ;
    delete iter ;
  }
}
