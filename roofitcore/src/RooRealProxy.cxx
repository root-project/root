/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.cc,v 1.6 2001/05/16 07:41:08 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooRealProxy is the concrete proxy for RooAbsReal objects
// A RooRealProxy is the general mechanism to store references
// to RooAbsReals inside a RooAbsArg
//
// RooRealProxy provides a cast operator to Double_t, allowing
// the proxy to functions a Double_t on the right hand side of expressions.

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
  if (!_arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
    cout << "RooRealProxy::min: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooAbsRealLValue*)_arg)->getFitMin() ;
}


Double_t RooRealProxy::max() const 
{
  // Return integration maximum for proxied real-valued object if it is a dependent
  if (!_arg->IsA()->InheritsFrom(RooAbsRealLValue::Class())) {
    cout << "RooRealProxy::max: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooAbsRealLValue*)_arg)->getFitMax() ;
}
