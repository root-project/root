/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategoryProxy.cc,v 1.1 2001/03/27 01:20:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooCategoryProxy.hh"

ClassImp(RooCategoryProxy)
;

RooCategoryProxy::RooCategoryProxy(const char* name, RooAbsArg* owner, RooAbsCategory& ref) : 
  RooArgProxy(name, owner,ref)
{
}


RooCategoryProxy::RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) : 
  RooArgProxy(name, owner, other) 
{
}


RooCategoryProxy::~RooCategoryProxy() 
{
}
