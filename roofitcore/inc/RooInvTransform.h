/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooInvTransform.rdl,v 1.5 2005/02/25 14:22:57 wverkerke Exp $
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
  inline virtual ~RooInvTransform() { }

  inline virtual Double_t operator()(const Double_t xvector[]) const {
    Double_t xinv= 1./xvector[0];
    return (*_func)(&xinv)*xinv*xinv;
  }
  inline virtual Double_t getMinLimit(UInt_t index) const { return 1/_func->getMaxLimit(index); }
  inline virtual Double_t getMaxLimit(UInt_t index) const { return 1/_func->getMinLimit(index); }

protected:
  const RooAbsFunc *_func;
  Double_t _scaleFactor;

  ClassDef(RooInvTransform,0) // RooAbsFunc decorator
};

#endif

