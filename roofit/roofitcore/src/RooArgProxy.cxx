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

#include "RooArgProxy.h"
#include "RooArgSet.h"
#include "RooAbsArg.h"
#include <iostream>
using std::ostream;

/**
\file RooArgProxy.cxx
\class RooArgProxy
\ingroup Roofitcore

Abstract interface for RooAbsArg proxy classes.
A RooArgProxy is the general mechanism to store references
to other RooAbsArgs inside a RooAbsArg.

Creating a RooArgProxy adds the proxied object to the proxy owners
server list (thus receiving value/shape dirty flags from it) and
registers itself with the owning class. The latter allows the
owning class to change the proxied pointer when the server it
points to gets redirected (e.g. in a copy or clone operation).
**/




////////////////////////////////////////////////////////////////////////////////
/// Constructor with owner and proxied variable.

RooArgProxy::RooArgProxy(const char* inName, const char* desc, RooAbsArg* owner,
          bool valueServer, bool shapeServer, bool proxyOwnsArg) :
  TNamed(inName,desc), _owner(owner),
  _valueServer(valueServer), _shapeServer(shapeServer), _ownArg(proxyOwnsArg)
{
  _owner->registerProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with owner and proxied variable. The valueServer and shapeServer booleans
/// control if the inserted client-server link in the owner propagates value and/or
/// shape dirty flags. If proxyOwnsArg is true, the proxy takes ownership of its component

RooArgProxy::RooArgProxy(const char *inName, const char *desc, RooAbsArg *owner, RooAbsArg &arg, bool valueServer,
                         bool shapeServer, bool proxyOwnsArg)
   : TNamed(inName, desc),
     _owner(owner),
     _arg(&arg),
     _valueServer(valueServer),
     _shapeServer(shapeServer),
     _isFund(_arg->isFundamental()),
     _ownArg(proxyOwnsArg)
{
  _owner->registerProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooArgProxy::RooArgProxy(const char* inName, RooAbsArg* owner, const RooArgProxy& other) :
  TNamed(inName,inName), RooAbsProxy(other), _owner(owner), _arg(other._arg),
  _valueServer(other._valueServer), _shapeServer(other._shapeServer),
  _isFund(other._isFund), _ownArg(other._ownArg)
{
  if (_ownArg) {
    _arg = _arg ? static_cast<RooAbsArg*>(_arg->Clone()) : nullptr ;
  }

  _owner->registerProxy(*this) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooArgProxy::~RooArgProxy()
{
  if (_owner) _owner->unRegisterProxy(*this) ;
  if (_ownArg) delete _arg ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change proxied object to object of same name in given list. If nameChange is true
/// the replacement object can have a different name and is identified as the replacement object by
/// the existence of a boolean attribute "origName:MyName" where MyName is the name of this instance

bool RooArgProxy::changePointer(const RooAbsCollection& newServerList, bool nameChange, bool factoryInitMode)
{
  RooAbsArg* newArg = nullptr;
  const bool initEmpty = _arg == nullptr;

  if (_arg) {
    newArg = _arg->findNewServer(newServerList, nameChange);
    if (newArg==_owner) newArg = nullptr;
  } else if (factoryInitMode) {
    newArg = newServerList.first() ;
    _owner->addServer(*newArg,_valueServer,_shapeServer) ;
  }

  if (newArg) {
    if (_ownArg) {
      // We refer to an object that somebody gave to us. Now, we are not owning it, any more.
      delete _arg;
      _ownArg = false;
    }

    _arg = newArg ;
    _isFund = _arg->isFundamental();
  }

  if (initEmpty && !factoryInitMode) return true;
  return newArg != nullptr;
}

bool RooArgProxy::changePointer(std::unordered_map<RooAbsArg *, RooAbsArg *> const &replacements)
{
   if (!_arg)
      return true;

   RooAbsArg *newArg = nullptr;

   auto newArgFound = replacements.find(_arg);
   if (newArgFound != replacements.end()) {
      newArg = newArgFound->second;
   }

   if (newArg) {
      if (_ownArg) {
         // We refer to an object that somebody gave to us. Now, we are not owning it, any more.
         delete _arg;
         _ownArg = false;
      }

      _arg = newArg;
      _isFund = _arg->isFundamental();
   }

   return newArg != nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Change the normalization set that should be offered to the
/// content objects getVal() when evaluated.

void RooArgProxy::changeDataSet(const RooArgSet* newNormSet)
{
  RooAbsProxy::changeNormSet(newNormSet) ;
  _arg->setProxyNormSet(newNormSet) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print the name of the proxy on ostream. If addContents is
/// true also the value of the contained RooAbsArg is also printed

void RooArgProxy::print(ostream& os, bool addContents) const
{
  os << name() << "=" << (_arg?_arg->GetName():"nullptr")  ;
  if (_arg && addContents) {
    os << "=" ;
    _arg->printStream(os,RooPrintable::kValue,RooPrintable::kInline) ;
  }
}
