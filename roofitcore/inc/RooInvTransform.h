/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooScaledFunc.rdl,v 1.1 2001/08/03 21:44:57 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_INV_TRANSFORM
#define ROO_INV_TRANSFORM

#include "RooFitCore/RooAbsFunc.hh"

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

