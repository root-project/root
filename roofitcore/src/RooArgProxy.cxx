/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgProxy.cc,v 1.7 2001/06/06 00:06:38 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsArg.hh"

// -- CLASS DESCRIPTION --
// RooArgProxy is the abstact interface for RooAbsArg proxy classes.
// A RooArgProxy is the general mechanism to store references
// to other RooAbsArgs inside a RooAbsArg
//
// Creating a RooArgProxy adds the proxied object to the proxy owners
// server list (thus receiving value/shape dirty flags from it) and
// registers itself with the owning class. The latter allows the
// owning class to change the proxied pointer when the server it
// points to gets redirected (e.g. in a copy or clone operation)


ClassImp(RooArgProxy)
;


RooArgProxy::RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg,
			 Bool_t valueServer, Bool_t shapeServer) : 
  TNamed(name,desc), _arg(&arg),
  _valueServer(valueServer), _shapeServer(shapeServer) 
{
  // Constructor with owner and proxied variable
  owner->registerProxy(*this) ;
}


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) : 
  RooAbsProxy(other), TNamed(other), _arg(other._arg)
{
  // Copy constructor
  owner->registerProxy(*this) ;
}


Bool_t RooArgProxy::changePointer(const RooArgSet& newServerList) 
{
  // Change proxied object to object of same name in given list
  RooAbsArg* newArg = newServerList.find(_arg->GetName()) ;
  if (newArg) _arg = newArg ;

  return newArg?kTRUE:kFALSE ;
}

void RooArgProxy::changeDataSet(const RooDataSet* newDataSet) 
{
  RooAbsProxy::changeDataSet(newDataSet) ;
  _arg->setProxyDataSet(newDataSet) ;
}
