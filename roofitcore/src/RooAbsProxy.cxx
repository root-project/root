/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsProxy.cc,v 1.1 2001/05/11 06:29:59 verkerke Exp $
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

ClassImp(RooAbsProxy)
;


RooAbsProxy::RooAbsProxy(const char* name, const char* desc, Bool_t valueServer, Bool_t shapeServer) : 
  TNamed(name,desc), _dset(0), 
  _valueServer(valueServer), _shapeServer(shapeServer) 
{
  // Constructor with owner and proxied object
}


RooAbsProxy::RooAbsProxy(const char* name, const RooAbsProxy& other) : 
  TNamed(other), _dset(other._dset),
  _valueServer(other._valueServer), _shapeServer(other._shapeServer) 
{
  // Copy constructor
  SetName(name) ;
}


void RooAbsProxy::changeDataSet(const RooDataSet* newDataSet) 
{
  // Destructor
  _dset = (RooDataSet*) newDataSet ;
}
