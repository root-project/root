/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.cc,v 1.1 2001/03/27 01:20:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooRealProxy.hh"

ClassImp(RooRealProxy)
;

RooRealProxy::RooRealProxy(const char* name, RooAbsArg* owner, RooAbsReal& ref) : 
  RooArgProxy(name, owner,ref)
{
}


RooRealProxy::RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) : 
  RooArgProxy(name, owner, other) 
{
}


RooRealProxy::~RooRealProxy() 
{
}
