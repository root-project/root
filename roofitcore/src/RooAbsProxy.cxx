/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsProxy.cc,v 1.3 2001/05/17 00:43:14 verkerke Exp $
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


RooAbsProxy::RooAbsProxy() : _dset(0)
{
  // Constructor with owner and proxied object
}


RooAbsProxy::RooAbsProxy(const char* name, const RooAbsProxy& other) : 
  _dset(other._dset)
{
  // Copy constructor
}


void RooAbsProxy::changeDataSet(const RooDataSet* newDataSet) 
{
  // Destructor
  _dset = (RooDataSet*) newDataSet ;
}
