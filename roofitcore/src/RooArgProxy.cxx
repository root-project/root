/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
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


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, RooAbsArg& arg) : 
  TNamed(name,name), _arg(&arg)
{
  owner->registerProxy(*this) ;
}


RooArgProxy::RooArgProxy(const char* name, RooAbsArg* owner, const RooArgProxy& other) : 
  TNamed(other), _arg(other._arg) 
{
  SetName(name) ;
  owner->registerProxy(*this) ;
}


Bool_t RooArgProxy::changePointer(RooArgSet& newServerList) 
{
  RooAbsArg* newArg = newServerList.find(_arg->GetName()) ;
  if (newArg) _arg = newArg ;

  return newArg?kTRUE:kFALSE ;
}
