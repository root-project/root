/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsProxy.cc,v 1.4 2001/06/06 00:06:38 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooAbsProxy.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooAbsArg.hh"

// -- CLASS DESCRIPTION --
// RooAbsProxy is the abstact interface for proxy classes.
// Proxy classes hold pointers to other RFC objects 
// and process serverRedirect changes so that the proxied
// pointers are updated accordingly on a clone or copy of
// of the owning class


ClassImp(RooAbsProxy)
;


RooAbsProxy::RooAbsProxy() : _nset(0)
{
  // Constructor with owner and proxied object
}


RooAbsProxy::RooAbsProxy(const char* name, const RooAbsProxy& other) : 
  _nset(other._nset)
{
  // Copy constructor
}


void RooAbsProxy::changeNormSet(const RooArgSet* newNormSet) 
{
  // Destructor
  _nset = (RooArgSet*) newNormSet ;
}
