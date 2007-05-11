/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvIntegrandBinding.rdl,v 1.2 2005/02/25 14:22:54 wverkerke Exp $
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
#ifndef ROO_CONV_INTEGRAND_BINDING
#define ROO_CONV_INTEGRAND_BINDING

#include "RooAbsFunc.h"

class RooAbsRealLValue;
class RooAbsReal;
class RooArgSet;

class RooConvIntegrandBinding : public RooAbsFunc {
public:
  RooConvIntegrandBinding(const RooAbsReal& func, const RooAbsReal& model, 
	             RooAbsReal& x, RooAbsReal& xprime, 
                     const RooArgSet* nset=0, Bool_t clipInvalid=kFALSE);
  virtual ~RooConvIntegrandBinding();

  virtual Double_t operator()(const Double_t xvector[]) const;
  virtual Double_t getMinLimit(UInt_t dimension) const;
  virtual Double_t getMaxLimit(UInt_t dimension) const;
  inline void setNormalizationSet(const RooArgSet* nset) { _nset = nset ; }

protected:
  void loadValues(const Double_t xvector[], Bool_t clipInvalid=kFALSE) const;

  const RooAbsReal *_func;
  const RooAbsReal *_model ;

  RooAbsRealLValue **_vars;
  const RooArgSet *_nset;
  mutable Bool_t _xvecValid;
  Bool_t _clipInvalid ;

  ClassDef(RooConvIntegrandBinding,0) // RooAbsReal interface adaptor
};

#endif

