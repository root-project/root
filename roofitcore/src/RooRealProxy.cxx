/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.cc,v 1.4 2001/05/11 06:30:01 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "TClass.h"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooRealVar.hh"

ClassImp(RooRealProxy)
;

RooRealProxy::RooRealProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsReal& ref,
			   Bool_t valueServer, Bool_t shapeServer) : 
  RooArgProxy(name, desc, owner,ref, valueServer, shapeServer)
{
  // Constructor with owner and proxied real-valued object
}


RooRealProxy::RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) : 
  RooArgProxy(name, owner, other) 
{
  // Copy constructor 
}


RooRealProxy::~RooRealProxy() 
{
  // Destructor
}


Double_t RooRealProxy::min() const 
{
  // Return integration minimum for proxied real-valued object if it is a dependent
  if (!_arg->IsA()->InheritsFrom(RooRealVar::Class())) {
    cout << "RooRealProxy::min: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooRealVar*)_arg)->getFitMin() ;
}


Double_t RooRealProxy::max() const 
{
  // Return integration maximum for proxied real-valued object if it is a dependent
  if (!_arg->IsA()->InheritsFrom(RooRealVar::Class())) {
    cout << "RooRealProxy::max: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooRealVar*)_arg)->getFitMax() ;
}
