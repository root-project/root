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
  inline RooScaledFunc(const RooAbsFunc &func, double scaleFactor) :
    RooAbsFunc(func.getDimension()), _func(&func), _scaleFactor(scaleFactor) { }
  inline ~RooScaledFunc() override { }

  inline double operator()(const double xvector[]) const override {
    return _scaleFactor*(*_func)(xvector);
  }
  inline double getMinLimit(UInt_t index) const override { return _func->getMinLimit(index); }
  inline double getMaxLimit(UInt_t index) const override { return _func->getMaxLimit(index); }

  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override {
    return _func->plotSamplingHint(obs,xlo,xhi) ;
  }

protected:
  const RooAbsFunc *_func;
  double _scaleFactor;

  ClassDefOverride(RooScaledFunc,0) // Function binding applying scaling to another function binding
};

#endif

