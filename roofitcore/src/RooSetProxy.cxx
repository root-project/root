/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooSetProxy.cc,v 1.1 2001/05/11 06:30:01 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
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


RooSetProxy::RooSetProxy(const char* name, const char* desc, RooAbsArg* owner, RooArgSet& set,
			 Bool_t valueServer, Bool_t shapeServer) : 
  RooAbsProxy(name,desc,valueServer,shapeServer), _set(&set)
{
  owner->registerProxy(*this) ;
}


RooSetProxy::RooSetProxy(const char* name, RooAbsArg* owner, const RooSetProxy& other) : 
  RooAbsProxy(name,other), _set(other._set)
{
  owner->registerProxy(*this) ;
}


Bool_t RooSetProxy::changePointer(const RooArgSet& newServerList) 
{
  TIterator* iter = _set->MakeIterator() ;
  RooAbsArg* arg ;
  Bool_t ok(kTRUE) ;
  while (arg=(RooAbsArg*)iter->Next()) {
    RooAbsArg* newArg = newServerList.find(arg->GetName()) ;
    if (newArg) ok |= _set->replace(*arg,*newArg) ;
  }

  return ok ;
}

