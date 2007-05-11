/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooArgProxy.cc,v 1.27 2005/06/16 09:31:26 wverkerke Exp $
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

#include "RooFit.h"

#include "RooArgProxy.h"
#include "RooArgProxy.h"
#include "RooArgSet.h"
#include "RooAbsArg.h"

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
  TNamed(name,desc), _owner(owner), _arg(&arg),
  _valueServer(valueServer), _shapeServer(shapeServer), _ownArg(proxyOwnsArg)
{
  // Constructor with owner and proxied variable
  _owner->registerProxy(*this) ;
  _isFund = _arg->isFundamental() ;
}


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) : 
  TNamed(name,name), RooAbsProxy(other), _owner(owner), _arg(other._arg), 
  _valueServer(other._valueServer), _shapeServer(other._shapeServer),
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
  if (_owner) _owner->unRegisterProxy(*this) ;
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
