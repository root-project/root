/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.rdl,v 1.2 2001/04/20 01:51:39 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_REAL_PROXY
#define ROO_REAL_PROXY

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgProxy.hh"

class RooRealProxy : public RooArgProxy {
public:

  // Constructors, assignment etc.
  RooRealProxy() {} ;
  RooRealProxy(const char* name, const char* desc, RooAbsArg* owner, RooAbsReal& ref) ;
  RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) ;
  virtual TObject* Clone(const char*) const { return new RooRealProxy(*this); }
  virtual ~RooRealProxy();

  // Accessors
  inline operator Double_t() const { return arg().getVal(_dset) ; }
  inline const RooAbsReal& arg() const { return (RooAbsReal&)*_arg ; }

  // Limits for integration
  Double_t min() const ;
  Double_t max() const ;

protected:

  ClassDef(RooRealProxy,0) // not persistable 
};

#endif
