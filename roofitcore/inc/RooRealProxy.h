/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealProxy.rdl,v 1.17 2004/11/29 12:22:23 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_REAL_PROXY
#define ROO_REAL_PROXY

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooAbsRealLValue.hh"

class RooRealProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooRealProxy() {} ;
  RooRealProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsReal& ref,
	       Bool_t valueServer=kTRUE, Bool_t shapeServer=kFALSE, Bool_t proxyOwnsArg=kFALSE) ;
  RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) ;
  virtual TObject* Clone(const char* newName=0) const { return new RooRealProxy(*this); }
  virtual ~RooRealProxy();

  // Accessors
  inline operator Double_t() const { return _isFund?((RooAbsReal*)_arg)->_value:((RooAbsReal*)_arg)->getVal(_nset) ; }
  inline const RooAbsReal& arg() const { return (RooAbsReal&)*_arg ; }

protected:

  inline RooAbsRealLValue* lvptr() const {
    // Assert that the held arg is an LValue
    RooAbsRealLValue* lvptr = (RooAbsRealLValue*)dynamic_cast<const RooAbsRealLValue*>(_arg) ;
    if (!lvptr) {
      std::cout << "RooRealProxy(" << name() << ")::INTERNAL error, expected " << _arg->GetName() << " to be an lvalue" << std::endl ;
      assert(0) ;
    }
    return lvptr ;
  }

public:

  // LValue operations 
  RooRealProxy& operator=(const Double_t& value) { lvptr()->setVal(value) ; return *this ; }
  Double_t min(const char* name=0) const { return lvptr()->getMin(name) ; }
  Double_t max(const char* name=0) const { return lvptr()->getMax(name) ; }


  ClassDef(RooRealProxy,0) // Proxy for a RooAbsReal object
};

#endif
