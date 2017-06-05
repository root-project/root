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
\file RooListProxy.cxx
\class RooListProxy
\ingroup Roofitcore

RooListProxy is the concrete proxy for RooArgList objects.
A RooListProxy is the only safe mechanism to store a RooArgList
with RooAbsArg contents in another RooAbsArg.
The list proxy has the semantic of a RooArgList but also 
takes care of all bookkeeping required when composite objects
are clone and client-server links need to be redirected.
**/


#include "RooFit.h"
#include "Riostream.h"

#include "RooListProxy.h"
#include "RooArgList.h"
#include "RooAbsArg.h"

using namespace std;

ClassImp(RooListProxy);
;



////////////////////////////////////////////////////////////////////////////////
/// Constructor with proxy name, description and pointer to ownder of
/// the RooListProxy. The default strategy for value/shape dirty flag
/// propagation of the list contents to the list owner is controlled
/// by the defValueServer and defShapeServer flags.

RooListProxy::RooListProxy(const char* inName, const char* /*desc*/, RooAbsArg* owner, 
			 Bool_t defValueServer, Bool_t defShapeServer) :
  RooArgList(inName), _owner(owner), 
  _defValueServer(defValueServer), 
  _defShapeServer(defShapeServer)
{
  _owner->registerProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor with name of proxy, pointer to owner of this proxy and
/// reference to list proxy to be copied

RooListProxy::RooListProxy(const char* inName, RooAbsArg* owner, const RooListProxy& other) : 
  RooArgList(other,inName), _owner(owner),  
  _defValueServer(other._defValueServer), 
  _defShapeServer(other._defShapeServer)
{
  _owner->registerProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooListProxy::~RooListProxy()
{
  if (_owner) _owner->unRegisterProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add object to list with explicitl directives on value and shape dirty flag propagation
/// of inserted object to list owner

Bool_t RooListProxy::add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent)
{
  Bool_t ret=RooArgList::add(var,silent) ;
  if (ret && _owner) {
    _owner->addServer((RooAbsArg&)var,valueServer,shapeServer) ;
  }
  return ret ;  
}



////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList::add()

Bool_t RooListProxy::add(const RooAbsArg& var, Bool_t silent) 
{
  return add(var,_defValueServer,_defShapeServer,silent) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList::addOwned()

Bool_t RooListProxy::addOwned(RooAbsArg& var, Bool_t silent)
{
  Bool_t ret=RooArgList::addOwned(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,_defValueServer,_defShapeServer) ;
  }
  return ret ;  
}


////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList::replace()

Bool_t RooListProxy::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  Bool_t ret=RooArgList::replace(var1,var2) ;
  if (ret) {
    _owner->removeServer((RooAbsArg&)var1) ;
    _owner->addServer((RooAbsArg&)var2,_owner->isValueServer(var1),
		                       _owner->isShapeServer(var2)) ;
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList::remove()

Bool_t RooListProxy::remove(const RooAbsArg& var, Bool_t silent, Bool_t matchByNameOnly) 
{
  Bool_t ret=RooArgList::remove(var,silent,matchByNameOnly) ;
  if (ret) {
    _owner->removeServer((RooAbsArg&)var) ;
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList::removeAll()

void RooListProxy::removeAll() 
{
  RooFIter iter = fwdIterator();
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter.next())) {
    _owner->removeServer(*arg) ;
  }

  RooArgList::removeAll() ;
}




////////////////////////////////////////////////////////////////////////////////
/// Reimplementation of standard RooArgList assignment operator

RooListProxy& RooListProxy::operator=(const RooArgList& other) 
{
  RooArgList::operator=(other) ;
  return *this ;
}




////////////////////////////////////////////////////////////////////////////////
/// Internal function that implements consequences of a server redirect on the
/// owner. If the list contains any element with names identical to those in newServerList
/// replace them with the instance in newServerList

Bool_t RooListProxy::changePointer(const RooAbsCollection& newServerList, Bool_t nameChange, Bool_t factoryInitMode) 
{
  if (getSize()==0) {
    if (factoryInitMode) {
      RooFIter iter = newServerList.fwdIterator() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)iter.next())) {
	if (arg!=_owner) {
	  add(*arg,kTRUE) ;
	}
      }
    } else {
      return kTRUE ;	
    }
  }
  RooFIter iter = fwdIterator();
  RooAbsArg* arg ;
  Bool_t error(kFALSE) ;
  while ((arg=(RooAbsArg*)iter.next())) {
    
    RooAbsArg* newArg= arg->findNewServer(newServerList, nameChange);
    if (newArg && newArg!=_owner) error |= !RooArgList::replace(*arg,*newArg) ;
  }
  return !error ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print the name of the proxy, and if requested a summary of
/// the contained elements as well

void RooListProxy::print(ostream& os, Bool_t addContents) const 
{ 
  if (!addContents) {
    os << name() << "=" ; printStream(os,kValue,kInline) ; 
  } else {
    os << name() << "=(" ;
    RooFIter iter = fwdIterator() ;
    RooAbsArg* arg ;
    Bool_t first2(kTRUE) ;
    while ((arg=(RooAbsArg*)iter.next())) {
      if (first2) {
	first2 = kFALSE ;
      } else {
	os << "," ;
      }
      arg->printStream(os,kValue|kName,kInline) ;
    }
    os << ")" ;
  }
}
