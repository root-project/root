/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCategoryProxy.cc,v 1.4 2001/05/11 06:30:00 verkerke Exp $
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

RooCategoryProxy::RooCategoryProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsCategory& ref,
				   Bool_t valueServer, Bool_t shapeServer) : 
  RooArgProxy(name, desc, owner, ref, valueServer, shapeServer)
{
  // Constructor with owner and proxied category object
}


RooCategoryProxy::RooCategoryProxy(const char* name, RooAbsArg* owner, const RooCategoryProxy& other) : 
  RooArgProxy(name, owner, other) 
{
  // Copy constructor
}


RooCategoryProxy::~RooCategoryProxy() 
{
  // Destructor
}
