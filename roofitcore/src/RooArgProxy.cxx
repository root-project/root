/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooArgProxy.cc,v 1.4 2001/05/11 06:29:59 verkerke Exp $
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

ClassImp(RooArgProxy)
;


RooArgProxy::RooArgProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsArg& arg,
			 Bool_t valueServer, Bool_t shapeServer) : 
  RooAbsProxy(name,desc,valueServer,shapeServer), _arg(&arg)
{
  // Constructor with owner and proxied variable
  owner->registerProxy(*this) ;
}


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) : 
  RooAbsProxy(name,other), _arg(other._arg)
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

