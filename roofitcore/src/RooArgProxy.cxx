/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgProxy.cc,v 1.15 2001/10/03 16:16:30 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
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
			 Bool_t valueServer, Bool_t shapeServer, Bool_t proxyOwnsArg) : 
  TNamed(name,desc), _arg(&arg),
  _valueServer(valueServer), _shapeServer(shapeServer), _owner(owner), _ownArg(proxyOwnsArg)
{
  // Constructor with owner and proxied variable
  _owner->registerProxy(*this) ;
  _isFund = _arg->isFundamental() ;
}


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) : 
  RooAbsProxy(other), TNamed(other), _arg(other._arg), 
  _valueServer(other._valueServer), _shapeServer(other._shapeServer), _owner(owner),
  _isFund(other._isFund), _ownArg(other._ownArg) 
{
  // Copy constructor

  if (_ownArg) {
    _arg = (RooAbsArg*) _arg->Clone() ;
  }

  _owner->registerProxy(*this) ;
}


RooArgProxy::~RooArgProxy()
{
  _owner->unRegisterProxy(*this) ;
  if (_ownArg) delete _arg ;
}


Bool_t RooArgProxy::changePointer(const RooAbsCollection& newServerList, Bool_t nameChange) 
{
  // Change proxied object to object of same name in given list

  RooAbsArg* newArg= _arg->findNewServer(newServerList, nameChange);
  if (newArg) {
    _arg = newArg ;
    _isFund = _arg->isFundamental() ;
  }

  return newArg?kTRUE:kFALSE ;
}

void RooArgProxy::changeDataSet(const RooArgSet* newNormSet) 
{
  RooAbsProxy::changeNormSet(newNormSet) ;
  _arg->setProxyNormSet(newNormSet) ;
}
