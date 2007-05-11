/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooScaledFunc.rdl,v 1.5 2005/02/25 14:23:02 wverkerke Exp $
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

class RooScaledFunc : public RooAbsFunc {
public:
  inline RooScaledFunc(const RooAbsFunc &func, Double_t scaleFactor) :
    RooAbsFunc(func.getDimension()), _func(&func), _scaleFactor(scaleFactor) { }
  inline virtual ~RooScaledFunc() { }

  inline virtual Double_t operator()(const Double_t xvector[]) const {
    return _scaleFactor*(*_func)(xvector);
  }
  inline virtual Double_t getMinLimit(UInt_t index) const { return _func->getMinLimit(index); }
  inline virtual Double_t getMaxLimit(UInt_t index) const { return _func->getMaxLimit(index); }

protected:
  const RooAbsFunc *_func;
  Double_t _scaleFactor;

  ClassDef(RooScaledFunc,0) // RooAbsFunc decorator
};

#endif

