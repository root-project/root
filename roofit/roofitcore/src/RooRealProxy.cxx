/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooRealProxy is the concrete proxy for RooAbsReal objects
// A RooRealProxy is the general mechanism to store references
// to RooAbsReals inside a RooAbsArg
//
// RooRealProxy provides a cast operator to Double_t, allowing
// the proxy to functions a Double_t on the right hand side of expressions.
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"
#include "RooRealProxy.h"
#include "RooRealVar.h"

using namespace std;

ClassImp(RooRealProxy)
;


//_____________________________________________________________________________
RooRealProxy::RooRealProxy(const char* inName, const char* desc, RooAbsArg* owner, 
			   Bool_t valueServer, Bool_t shapeServer, Bool_t ownArg) : 
  RooArgProxy(inName, desc, owner, valueServer, shapeServer, ownArg)
{
  // Constructor with owner. 
}



//_____________________________________________________________________________
RooRealProxy::RooRealProxy(const char* inName, const char* desc, RooAbsArg* owner, RooAbsReal& ref,
			   Bool_t valueServer, Bool_t shapeServer, Bool_t ownArg) : 
  RooArgProxy(inName, desc, owner,ref, valueServer, shapeServer, ownArg)
{
  // Constructor with owner and proxied real-valued object. The propagation
  // of value and shape dirty flags of the contained arg to the owner is
  // controlled by the valueServer and shapeServer flags. If ownArg is true
  // the proxy will take ownership of the contained arg
}



//_____________________________________________________________________________
RooRealProxy::RooRealProxy(const char* inName, RooAbsArg* owner, const RooRealProxy& other) : 
  RooArgProxy(inName, owner, other) 
{
  // Copy constructor 
}



//_____________________________________________________________________________
RooRealProxy::~RooRealProxy() 
{
  // Destructor
}



//_____________________________________________________________________________
RooAbsRealLValue* RooRealProxy::lvptr() const 
{
  // Return l-value pointer to contents, if contents is in fact an l-value

  // WVE remove check here -- need to put it back in setArg and ctor
  return (RooAbsRealLValue*) _arg ;

  // Assert that the held arg is an LValue
  RooAbsRealLValue* Lvptr = (RooAbsRealLValue*)dynamic_cast<const RooAbsRealLValue*>(_arg) ;
  if (!Lvptr) {
    cout << "RooRealProxy(" << name() << ")::INTERNAL error, expected " << _arg->GetName() << " to be an lvalue" << endl ;
    assert(0) ;
  }
  return Lvptr ;
}


//_____________________________________________________________________________
Bool_t RooRealProxy::setArg(RooAbsReal& newRef) 
{
  // Change object held in proxy into newRef
  if (absArg()) {
    if (TString(arg().GetName()!=newRef.GetName())) {
      newRef.setAttribute(Form("ORIGNAME:%s",arg().GetName())) ;
    }
    return changePointer(RooArgSet(newRef),kTRUE) ;
  } else {
    return changePointer(RooArgSet(newRef),kFALSE,kTRUE);
  }
}
