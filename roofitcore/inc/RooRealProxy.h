/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealProxy.rdl,v 1.1 2001/03/27 01:20:20 verkerke Exp $
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
  RooRealProxy(const char* name, RooAbsArg* owner, RooAbsReal& ref) ;
  RooRealProxy(const char* name, RooAbsArg* owner, const RooRealProxy& other) ;
  virtual TObject* Clone() { return new RooRealProxy(*this); }
  virtual ~RooRealProxy();
  

  // Accessors
  inline operator Double_t() const { return arg().getVal() ; }
  inline const RooAbsReal& arg() const { return (RooAbsReal&)*_arg ; }

protected:

  ClassDef(RooRealProxy,0) // not persistable 
};

#endif
