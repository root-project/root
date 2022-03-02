/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooScaledFunc.h,v 1.6 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SCALED_FUNC
#define ROO_SCALED_FUNC

#include "RooAbsFunc.h"
#include <list>

class RooScaledFunc : public RooAbsFunc {
public:
  inline RooScaledFunc(const RooAbsFunc &func, Double_t scaleFactor) :
    RooAbsFunc(func.getDimension()), _func(&func), _scaleFactor(scaleFactor) { }
  inline ~RooScaledFunc() override { }

  inline Double_t operator()(const Double_t xvector[]) const override {
    return _scaleFactor*(*_func)(xvector);
  }
  inline Double_t getMinLimit(UInt_t index) const override { return _func->getMinLimit(index); }
  inline Double_t getMaxLimit(UInt_t index) const override { return _func->getMaxLimit(index); }

  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const override {
    return _func->plotSamplingHint(obs,xlo,xhi) ;
  }

protected:
  const RooAbsFunc *_func;
  Double_t _scaleFactor;

  ClassDefOverride(RooScaledFunc,0) // Function binding applying scaling to another function binding
};

#endif

