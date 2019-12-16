/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealProxy.h,v 1.23 2007/07/12 20:30:28 wouter Exp $
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
#ifndef ROO_TEMPLATE_PROXY
#define ROO_TEMPLATE_PROXY

#include "RooAbsReal.h"
#include "RooArgProxy.h"
#include "RooAbsRealLValue.h"

/**
\class RooTemplateProxy
\ingroup Roofitcore

A RooTemplateProxy is used to hold references to other objects in an expression tree.
A `RooGaussian(..., x, mean, sigma)` e.g. stores `x, mean, sigma` as `RooRealProxy (=
RooTemplateProxy<RooAbsReal>`.

This allows access to their current values, to retrieve batches of observable data, and it
automatically registers these variables as "servers" of the Gaussian.

Renaming or exchanging objects that serve values to the owner of the proxy is handled automatically.

A few typedefs have been defined:
- `RooRealProxy = RooTemplateProxy<RooAbsReal>`. Any generic object that converts to a real value.
- `RooPdfProxy  = RooTemplateProxy<RooAbsPdf>`.  Handle to PDFs.
- `RooLVarProxy = RooTemplateProxy<RooAbsRealLValue>`. Handle to real values that can be assigned to.
- `RooRealVarProxy = RooTemplateProxy<RooRealVar>`. Handle to RooRealVars.
**/

template<class T>
class RooTemplateProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooTemplateProxy() {} ;

  ////////////////////////////////////////////////////////////////////////////////
  /// Constructor with owner.
  RooTemplateProxy(const char* theName, const char* desc, RooAbsArg* owner,
      Bool_t valueServer=true, Bool_t shapeServer=false, Bool_t proxyOwnsArg=false)
  : RooArgProxy(theName, desc, owner, valueServer, shapeServer, proxyOwnsArg) { }

  ////////////////////////////////////////////////////////////////////////////////
  /// Constructor with owner and proxied real-valued object. The propagation
  /// of value and shape dirty flags of the contained arg to the owner is
  /// controlled by the valueServer and shapeServer flags. If ownArg is true
  /// the proxy will take ownership of the contained arg.
  RooTemplateProxy(const char* theName, const char* desc, RooAbsArg* owner, T& ref,
      Bool_t valueServer=true, Bool_t shapeServer=false, Bool_t proxyOwnsArg=false) :
        RooArgProxy(theName, desc, owner, ref, valueServer, shapeServer, proxyOwnsArg) { }


  ////////////////////////////////////////////////////////////////////////////////
  /// Copy constructor.
  RooTemplateProxy(const char* theName, RooAbsArg* owner, const RooTemplateProxy& other) :
    RooArgProxy(theName, owner, other) { }

  virtual TObject* Clone(const char* newName=0) const { return new RooTemplateProxy<T>(newName,_owner,*this); }

  inline operator Double_t() const {
    return arg().getVal(_nset);
  }

  RooSpan<const double> getValBatch(std::size_t begin, std::size_t batchSize) const {
    return arg().getValBatch(begin, batchSize, _nset);
  }

  /// Return reference to object held in proxy.
  inline const T& arg() const { return static_cast<T&>(*_arg); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Change object held in proxy into newRef
  Bool_t setArg(T& newRef) {
    if (absArg()) {
      if (TString(arg().GetName()!=newRef.GetName())) {
        newRef.setAttribute(Form("ORIGNAME:%s",arg().GetName())) ;
      }
      return changePointer(RooArgSet(newRef),kTRUE) ;
    } else {
      return changePointer(RooArgSet(newRef),kFALSE,kTRUE);
    }
  }


  /// Assign a new value to the object pointed to by the proxy. This requires the payload to be assignable (RooAbsRealLValue or derived).
  RooTemplateProxy<T>& operator=(const Double_t& value) { lvptr()->setVal(value) ; return *this ; }
  /// Query lower limit of range. This requires the payload to be RooAbsRealLValue or derived.
  Double_t min(const char* rname=0) const { return lvptr()->getMin(rname) ; }
  /// Query upper limit of range. This requires the payload to be RooAbsRealLValue or derived.
  Double_t max(const char* rname=0) const { return lvptr()->getMax(rname) ; }
  /// Check if the range has a lower bound. This requires the payload to be RooAbsRealLValue or derived.
  Bool_t hasMin(const char* rname=0) const { return lvptr()->hasMin(rname) ; }
  /// Check if the range has a upper bound. This requires the payload to be RooAbsRealLValue or derived.
  Bool_t hasMax(const char* rname=0) const { return lvptr()->hasMax(rname) ; }


private:
  ////////////////////////////////////////////////////////////////////////////////
  /// Return l-value pointer to contents. If the contents derive from RooAbsLValue, the function
  /// simply returns the pointer.
  /// If the template parameter of this proxy does not derive from RooAbsLValue, then
  /// - in a debug build, a dynamic_cast is attempted.
  /// - in a release build, a static_cast is forced, irrespective of what the type of the object actually is. This
  /// is dangerous, but equivalent to the behaviour before refactoring.
  /// \deprecated This function is unneccessary if the template parameter is RooAbsRealLValue or a
  /// derived type, as arg() will always return the correct type.
  RooAbsRealLValue* lvptr() const {
    return lvptr_impl(static_cast<T*>(nullptr));
  }

  /// Overload with RooAbsRealLValue and derived types. Just returns the pointer.
  RooAbsRealLValue* lvptr_impl(RooAbsRealLValue*) const {
    return _arg;
  }

  /// Overload with base types. Attempts a cast.
  /// \deprecated The payload of this proxy should be at least RooAbsRealLValue or more derived for
  /// this function to be called safely.
  RooAbsRealLValue* lvptr_impl(RooAbsArg*) const
    R__SUGGEST_ALTERNATIVE("The template argument of RooTemplateProxy needs to derive from RooAbsRealLValue.") {
#ifdef NDEBUG
    return static_cast<RooAbsRealLValue*>(_arg);
#else
    auto arg = dynamic_cast<RooAbsRealLValue*>(_arg);
    assert(arg);
    return arg;
#endif
  }

  ClassDef(RooTemplateProxy,1) // Proxy for a RooAbsReal object
};

/// Compatibility typedef for old uses of this proxy.
using RooRealProxy = RooTemplateProxy<RooAbsReal>;
using RooPdfProxy     = RooTemplateProxy<RooAbsPdf>;
using RooLVarProxy    = RooTemplateProxy<RooAbsRealLValue>;
using RooRealVarProxy = RooTemplateProxy<RooRealVar>;

#endif
