/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealBinding.rdl,v 1.8 2005/02/25 14:23:01 wverkerke Exp $
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
#ifndef ROO_REAL_BINDING
#define ROO_REAL_BINDING

#include "RooAbsFunc.h"

class RooAbsRealLValue;
class RooAbsReal;
class RooArgSet;

class RooRealBinding : public RooAbsFunc {
public:
  RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset=0, Bool_t clipInvalid=kFALSE, const TNamed* rangeName=0);
  virtual ~RooRealBinding();

  virtual Double_t operator()(const Double_t xvector[]) const;
  virtual Double_t getMinLimit(UInt_t dimension) const;
  virtual Double_t getMaxLimit(UInt_t dimension) const;

protected:
  void loadValues(const Double_t xvector[]) const;
  const RooAbsReal *_func;
  RooAbsRealLValue **_vars;
  const RooArgSet *_nset;
  mutable Bool_t _xvecValid;
  Bool_t _clipInvalid ;
  const TNamed* _rangeName ; //!

  ClassDef(RooRealBinding,0) // RooAbsReal interface adaptor
};

#endif

