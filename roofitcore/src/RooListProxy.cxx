/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooListProxy.cc,v 1.6 2001/10/22 07:12:13 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   04-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [CONT] --
// RooListProxy is the concrete proxy for RooArgSet objects.
// A RooListProxy is the general mechanism to store a RooArgSet
// with RooAbsArgs in a RooAbsArg.
//
// Creating a RooListProxy adds all members of the proxied RooArgSet to the proxy owners
// server list (thus receiving value/shape dirty flags from it) and
// registers itself with the owning class. The latter allows the
// owning class to update the pointers of RooArgSet contents to reflect
// the serverRedirect changes.


#include "RooFitCore/RooListProxy.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooAbsArg.hh"

ClassImp(RooListProxy)
;


RooListProxy::RooListProxy(const char* name, const char* desc, RooAbsArg* owner, 
			 Bool_t defValueServer, Bool_t defShapeServer) :
  RooArgList(name), _owner(owner), 
  _defValueServer(defValueServer), 
  _defShapeServer(defShapeServer)
{
  //SetTitle(desc) ;
  _owner->registerProxy(*this) ;
  _iter = createIterator() ;
}


RooListProxy::RooListProxy(const char* name, RooAbsArg* owner, const RooListProxy& other) : 
  RooArgList(other,name), _owner(owner),  
  _defValueServer(other._defValueServer), 
  _defShapeServer(other._defShapeServer)
{
  _owner->registerProxy(*this) ;
  _iter = createIterator() ;
}


RooListProxy::~RooListProxy()
{
  _owner->unRegisterProxy(*this) ;
  delete _iter ;
}


Bool_t RooListProxy::add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent)
{
  Bool_t ret=RooArgList::add(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,valueServer,shapeServer) ;
  }
  return ret ;  
}


Bool_t RooListProxy::add(const RooAbsArg& var, Bool_t silent) 
{
  return add(var,_defValueServer,_defShapeServer,silent) ;
}


Bool_t RooListProxy::addOwned(RooAbsArg& var, Bool_t silent)
{
  Bool_t ret=RooArgList::addOwned(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,_defValueServer,_defShapeServer) ;
  }
  return ret ;  
}

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


Bool_t RooListProxy::remove(const RooAbsArg& var, Bool_t silent, Bool_t matchByNameOnly) 
{
  Bool_t ret=RooArgList::remove(var,silent,matchByNameOnly) ;
  if (ret) {
    _owner->removeServer((RooAbsArg&)var) ;
  }
  return ret ;
}


void RooListProxy::removeAll() 
{
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _owner->removeServer(*arg) ;
  }
  delete iter ;

  RooArgList::removeAll() ;
}




RooListProxy& RooListProxy::operator=(const RooArgList& other) 
{
  RooArgList::operator=(other) ;
  return *this ;
}




Bool_t RooListProxy::changePointer(const RooAbsCollection& newServerList, Bool_t nameChange) 
{
  if (getSize()==0) return kTRUE ;

  _iter->Reset() ;
  RooAbsArg* arg ;
  Bool_t error(kFALSE) ;
  while (arg=(RooAbsArg*)_iter->Next()) {
    
    RooAbsArg* newArg= arg->findNewServer(newServerList, nameChange);
    if (newArg) error |= !RooArgList::replace(*arg,*newArg) ;
  }
  return !error ;
}

