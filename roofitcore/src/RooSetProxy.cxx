/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSetProxy.cc,v 1.3 2001/06/06 00:06:39 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   04-Jun-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooSetProxy is the concrete proxy for RooArgSet objects.
// A RooSetProxy is the general mechanism to store a RooArgSet
// with RooAbsArgs in a RooAbsArg.
//
// Creating a RooSetProxy adds all members of the proxied RooArgSet to the proxy owners
// server list (thus receiving value/shape dirty flags from it) and
// registers itself with the owning class. The latter allows the
// owning class to update the pointers of RooArgSet contents to reflect
// the serverRedirect changes.


#include "RooFitCore/RooSetProxy.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsArg.hh"

ClassImp(RooSetProxy)
;


RooSetProxy::RooSetProxy(const char* name, const char* desc, RooAbsArg* owner, 
			 Bool_t defValueServer, Bool_t defShapeServer) :
  RooArgSet(name), _owner(owner), 
  _defValueServer(defValueServer), 
  _defShapeServer(defShapeServer)
{
  //SetTitle(desc) ;
  _owner->registerProxy(*this) ;
}


RooSetProxy::RooSetProxy(const char* name, RooAbsArg* owner, const RooSetProxy& other) : 
  RooArgSet(other,name), _owner(owner),  
  _defValueServer(other._defValueServer), 
  _defShapeServer(other._defShapeServer)
{
  _owner->registerProxy(*this) ;
}


Bool_t RooSetProxy::add(const RooAbsArg& var, Bool_t valueServer, Bool_t shapeServer, Bool_t silent)
{
  Bool_t ret=RooArgSet::add(var,silent) ;
  if (ret) {
    _owner->addServer((RooAbsArg&)var,valueServer,shapeServer) ;
  }
  return ret ;  
}


Bool_t RooSetProxy::add(const RooAbsArg& var, Bool_t silent) 
{
  return add(var,_defValueServer,_defShapeServer,silent) ;
}


Bool_t RooSetProxy::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  Bool_t ret=RooArgSet::replace(var1,var2) ;
  if (ret) {
    _owner->removeServer((RooAbsArg&)var1) ;
    _owner->addServer((RooAbsArg&)var2,_owner->isValueServer(var1),
		                       _owner->isShapeServer(var2)) ;
  }
  return ret ;
}


Bool_t RooSetProxy::remove(const RooAbsArg& var) 
{
  Bool_t ret=RooArgSet::remove(var) ;
  if (ret) {
    _owner->removeServer((RooAbsArg&)var) ;
  }
  return ret ;
}


void RooSetProxy::removeAll() 
{
  TIterator* iter = MakeIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    _owner->removeServer(*arg) ;
  }
  delete iter ;
}




RooSetProxy& RooSetProxy::operator=(const RooArgSet& other) 
{
  RooArgSet::operator=(other) ;
  return *this ;
}




Bool_t RooSetProxy::changePointer(const RooArgSet& newServerList) 
{
  TIterator* iter = MakeIterator() ;
  RooAbsArg* arg ;
  Bool_t error(kFALSE) ;
  while (arg=(RooAbsArg*)iter->Next()) {
    RooAbsArg* newArg = newServerList.find(arg->GetName()) ;
    if (newArg) error |= !RooArgSet::replace(*arg,*newArg) ;
  }
  delete iter ;
  return !error ;
}

