/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.cc,v 1.2 2001/04/20 01:51:39 verkerke Exp $
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

RooRealProxy::RooRealProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsReal& ref) : 
  RooArgProxy(name, desc, owner,ref)
{
}


RooRealProxy::RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) : 
  RooArgProxy(name, owner, other) 
{
}


RooRealProxy::~RooRealProxy() 
{
}


Double_t RooRealProxy::min() const 
{
  if (!_arg->IsA()->InheritsFrom(RooRealVar::Class())) {
    cout << "RooRealProxy::min: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooRealVar*)_arg)->getFitMin() ;
}


Double_t RooRealProxy::max() const 
{
  if (!_arg->IsA()->InheritsFrom(RooRealVar::Class())) {
    cout << "RooRealProxy::max: INTERNAL error, expected " << _arg->GetName() << " to be a RooRealVar" << endl ;
    assert(0) ;
  }
  return ((RooRealVar*)_arg)->getFitMax() ;
}
