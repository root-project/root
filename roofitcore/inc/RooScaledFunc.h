/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsFunc1D.rdl,v 1.2 2001/05/14 22:54:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_SCALED_FUNC
#define ROO_SCALED_FUNC

#include "RooFitCore/RooAbsFunc.hh"

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

