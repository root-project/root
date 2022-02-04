/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooInvTransform.h,v 1.6 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_INV_TRANSFORM
#define ROO_INV_TRANSFORM

#include "RooAbsFunc.h"

class RooInvTransform : public RooAbsFunc {
public:
  RooInvTransform(const RooAbsFunc &func);
  inline ~RooInvTransform() override { }

  inline Double_t operator()(const Double_t xvector[]) const override {
    Double_t xinv= 1./xvector[0];
    return (*_func)(&xinv)*xinv*xinv;
  }
  inline Double_t getMinLimit(UInt_t index) const override { return 1/_func->getMaxLimit(index); }
  inline Double_t getMaxLimit(UInt_t index) const override { return 1/_func->getMinLimit(index); }

protected:
  const RooAbsFunc *_func; ///< Input function binding

  ClassDefOverride(RooInvTransform,0) // Function binding returning inverse of other function binding
};

#endif

